import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.esm_block import DihedralFeatures
from src.layers import GVP, GVPConvLayer, LayerNorm, MultiGVPConvLayer
from src.utils import register_model


@register_model(name="GVPDiff")
class GVPDiff(torch.nn.Module):
    """GVP diffusion model for RNA design."""

    def __init__(self, config):
        super().__init__()
        self.node_in_dim = tuple(config.node_in_dim)
        self.node_h_dim = tuple(config.node_h_dim)
        self.edge_in_dim = tuple(config.edge_in_dim)
        self.edge_h_dim = tuple(config.edge_h_dim)
        self.edge_model_dim = self.edge_in_dim
        self.num_layers = config.num_layers
        self.out_dim = config.out_dim
        self.time_cond = config.time_cond

        activations = (F.silu, None)
        drop_rate = config.drop_rate
        self.embed_dihedral = DihedralFeatures(self.node_h_dim[0])

        self.W_v = nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True),
        )

        self.W_e = nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_model_dim, activations=(None, None), vector_gate=True),
        )

        self.encoder_layers = nn.ModuleList(
            MultiGVPConvLayer(
                self.node_h_dim,
                self.edge_model_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                norm_first=True,
            )
            for _ in range(self.num_layers)
        )

        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))
        self.dec_in_proj = nn.Linear(self.out_dim + self.node_h_dim[0], self.node_h_dim[0])

        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(
                self.node_h_dim,
                self.edge_model_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                autoregressive=False,
            )
            for _ in range(self.num_layers)
        )

        if self.time_cond:
            learned_sinu_dim = 16
            time_cond_dim = self.node_h_dim[0] * 2
            self.to_time_hiddens = nn.Sequential(
                LearnedSinusoidalPosEmb(learned_sinu_dim),
                nn.Linear(learned_sinu_dim + 1, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, self.node_h_dim[0]),
            )

    def _encode_graph(self, batch):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        return h_V, h_E, edge_index

    def _time_embedding(self, noise_level, n_samples, device):
        if not self.time_cond or noise_level is None:
            return torch.zeros(max(n_samples, 1), self.node_h_dim[0], device=device)

        t_emb = self.to_time_hiddens(noise_level)
        t_emb = t_emb.view(-1, t_emb.size(-1))
        rows = t_emb.shape[0]
        if rows < n_samples:
            t_emb = t_emb.repeat(int(math.ceil(n_samples / rows)), 1)
        return t_emb[:n_samples]

    def _prepare_decoder_hidden(self, z, noise_level, n_samples, num_nodes, device):
        z_flat = z.reshape(n_samples * num_nodes, -1)
        if self.time_cond:
            t_emb = self._time_embedding(noise_level, n_samples, device)
            t_emb = t_emb.unsqueeze(1).expand(-1, num_nodes, -1).reshape(n_samples * num_nodes, -1)
        else:
            t_emb = torch.zeros(n_samples * num_nodes, self.node_h_dim[0], device=device)

        dec_input = torch.cat([z_flat, t_emb], dim=-1)
        return self.dec_in_proj(dec_input)

    def forward(self, batch, time=None, noise_level=None, **kwargs):
        length = batch.seq.shape[0]
        z_t = batch.z_t

        encoder_embeddings, h_E, edge_index = self._encode_graph(batch)
        dec_hidden = self._prepare_decoder_hidden(z_t, noise_level, 1, length, z_t.device).view(length, -1)

        h_V_dec = (encoder_embeddings[0] + dec_hidden, encoder_embeddings[1])
        for layer in self.decoder_layers:
            h_V_dec = layer(h_V_dec, edge_index, h_E, autoregressive_x=None)

        pred_noise = self.W_out(h_V_dec)
        return pred_noise.unsqueeze(0)

    def sample(
        self,
        data,
        n_samples: int,
        time: Tensor = None,
        noise_level: Tensor = None,
        **kwargs,
    ) -> Tensor:
        num_nodes = data.seq.size(0)

        encoder_embeddings, h_E, edge_index = self._encode_graph(data)

        s_h, v_h = encoder_embeddings
        s_h = s_h.unsqueeze(0).expand(n_samples, -1, -1)
        v_h = v_h.unsqueeze(0).expand(n_samples, -1, -1, -1)
        total_nodes = n_samples * num_nodes
        s_h = s_h.reshape(total_nodes, -1)
        v_h = v_h.reshape(total_nodes, v_h.size(2), 3)
        h_V_batch = (s_h, v_h)

        s_e, v_e = h_E
        s_e = s_e.unsqueeze(0).expand(n_samples, -1, -1).reshape(-1, s_e.size(1))
        v_e = v_e.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(-1, v_e.size(2), 3)
        h_E_batch = (s_e, v_e)

        edge_indices = [edge_index + i * num_nodes for i in range(n_samples)]
        batch_edge_index = torch.cat(edge_indices, dim=1)

        z = data.z_t
        dec_hidden = self._prepare_decoder_hidden(z, noise_level, n_samples, num_nodes, data.seq.device)

        h_s = h_V_batch[0] + dec_hidden
        h_v = h_V_batch[1]
        h_V_dec = (h_s, h_v)

        for layer in self.decoder_layers:
            h_V_dec = layer(h_V_dec, batch_edge_index, h_E_batch, autoregressive_x=None)
        out = self.W_out(h_V_dec)
        if isinstance(out, tuple):
            out = out[0]

        return out.view(n_samples, num_nodes, self.out_dim)

    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        if mask_confs.size(1) == 1:
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])

        n_conf_true = mask_confs.sum(1, keepdim=True)

        mask = mask_confs.unsqueeze(2)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        mask = mask.unsqueeze(3)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]

        h_V = (h_V0.sum(dim=1) / n_conf_true, h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))
        h_E = (
            h_E0.sum(dim=1) / n_conf_true[edge_index[0]],
            h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2),
        )
        return h_V, h_E


class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)
