import os
import shutil
from datetime import datetime

import numpy as np
import wandb

import torch

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd

from src.data.data_utils import pdb_to_tensor, get_c4p_coords
from src.constants import NUM_TO_LETTER, PROJECT_PATH, RMSD_THRESHOLD, TM_THRESHOLD, GDT_THRESHOLD


def evaluate(
    rhofold,
    dataset,
    raw_data,
    pred_seq,
    n_samples,
    device,
    save_designs=False,
    parallel_id=None,
):
    """Evaluate a predicted sequence with RhoFold-based tertiary metrics."""
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results = {"samples_list": []}

    with torch.no_grad():
        data = dataset.featurizer(raw_data).to(device)
        samples = pred_seq.unsqueeze(0)
        results["samples_list"].append(samples.cpu().numpy())

        mask_coords = data.mask_coords.cpu().numpy()
        try:
            output_dir = os.path.join(wandb.run.dir, f"designs_eval/{current_datetime}/sample0/")
        except AttributeError:
            output_dir = os.path.join(PROJECT_PATH, f"designs_eval/{current_datetime}/sample0/")

        sc_score_rmsd, sc_score_tm, sc_score_gdt = self_consistency_score_rhofold(
            samples.cpu().numpy(),
            raw_data,
            mask_coords,
            rhofold,
            output_dir,
            save_designs=save_designs,
            parallel_id=parallel_id,
        )

        results["sc_score_rmsd"] = [sc_score_rmsd.mean()]
        results["sc_score_tm"] = [sc_score_tm.mean()]
        results["sc_score_gddt"] = [sc_score_gdt.mean()]
        results["rmsd_within_thresh"] = [(sc_score_rmsd <= RMSD_THRESHOLD).sum() / n_samples]
        results["tm_within_thresh"] = [(sc_score_tm >= TM_THRESHOLD).sum() / n_samples]
        results["gddt_within_thresh"] = [(sc_score_gdt >= GDT_THRESHOLD).sum() / n_samples]

    return results


def self_consistency_score_rhofold(
    samples,
    true_raw_data,
    mask_coords,
    rhofold,
    output_dir,
    num_to_letter=NUM_TO_LETTER,
    save_designs=False,
    save_pdbs=False,
    use_relax=False,
    parallel_id=None,
):
    """Compute RMSD/TM/GDT between RhoFold predictions and the reference structure."""
    os.makedirs(output_dir, exist_ok=True)

    input_seq = SeqRecord(Seq(true_raw_data["sequence"]), id="input_sequence,", description="input_sequence")
    sequences = [input_seq]

    sc_rmsds = []
    sc_tms = []
    sc_gddts = []

    for seq in samples:
        idx = (parallel_id + 1) if parallel_id is not None else 0

        seq = SeqRecord(
            Seq("".join([num_to_letter[num] for num in seq])),
            id=f"sample={idx},",
            description=f"sample={idx}",
        )
        sequences.append(seq)
        design_fasta_path = os.path.join(output_dir, f"design{idx}.fasta")
        SeqIO.write(seq, design_fasta_path, "fasta")

        design_pdb_path = os.path.join(output_dir, f"design{idx}.pdb")
        rhofold.predict(design_fasta_path, design_pdb_path, use_relax)

        _, coords, _, _ = pdb_to_tensor(
            design_pdb_path,
            return_sec_struct=False,
            return_sasa=False,
            keep_insertions=False,
        )
        coords = get_c4p_coords(coords)
        coords = coords - coords.mean(dim=0)

        if coords.shape[0] == mask_coords.shape[0]:
            coords = coords[mask_coords, :]

        other_coords = true_raw_data["coords_list"][0]
        if other_coords.shape[0] == mask_coords.shape[0]:
            ref = get_c4p_coords(other_coords)[mask_coords, :]
        else:
            ref = get_c4p_coords(other_coords)
        ref = ref - ref.mean(dim=0)

        rot = rotation_matrix(ref, coords)[0]
        ref = ref @ rot.T

        sc_rmsds.append(get_rmsd(coords, ref, superposition=True, center=True))
        sc_tms.append(get_tmscore(coords, ref))
        sc_gddts.append(get_gddt(coords, ref))

        if os.path.exists(design_fasta_path):
            os.unlink(design_fasta_path)
        if not save_pdbs and os.path.exists(design_pdb_path):
            os.unlink(design_pdb_path)

    if save_designs:
        SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")
    elif os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    return np.array(sc_rmsds), np.array(sc_tms), np.array(sc_gddts)


def get_tmscore(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute TM-score from aligned C4' coordinates."""
    l_target = y.shape[0]
    d0_l_target = 1.24 * np.power(l_target - 15, 1 / 3) - 1.8
    di = torch.pairwise_distance(y_hat, y)
    out = torch.sum(1 / (1 + (di / d0_l_target) ** 2)) / l_target
    if torch.isnan(out):
        return torch.tensor(0.0)
    return out


def get_gddt(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute GDT-TS from aligned C4' coordinates."""
    dist = torch.norm(y - y_hat, dim=1)

    count_1 = (dist < 1).sum() / dist.numel()
    count_2 = (dist < 2).sum() / dist.numel()
    count_4 = (dist < 4).sum() / dist.numel()
    count_8 = (dist < 8).sum() / dist.numel()
    out = torch.mean(torch.tensor([count_1, count_2, count_4, count_8]))
    if torch.isnan(out):
        return torch.tensor(0.0)
    return out
