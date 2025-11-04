import gc
import json
import os
import random
import subprocess
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py3Dmol
import torch
from prody import parsePDB
import gemmi

from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_molecules, load_canonicals
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import (
    MSA,
    Coords,
    Ensemble,
    Input,
    Structure,
    StructureV2,
)
from boltz.data.write.pdb import to_pdb
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBIO import PDBIO
import logging

# Existing filter
warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
warnings.filterwarnings(
    "ignore",
    message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*",
    category=UserWarning,
)

alphabet = list[str]("XXARNDCQEGHILKMFPSTWYV-")

chain_to_number = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
}


def get_cif(cif_code=""):
    """
    Returns the local filename (relative path) to the CIF for the provided code or file.
    Checks local directory and fetches from RCSB or AlphaFold if needed.
    """
    if cif_code is None or cif_code == "":
        print("Error: No cif code specified and uploads not supported in CLI mode.")
        sys.exit(1)
    elif os.path.isfile(cif_code):
        return os.path.abspath(cif_code)
    elif len(cif_code) == 4:
        local_cif = f"{cif_code}.cif"
        if not os.path.isfile(local_cif):
            os.system(f"wget -qnc https://files.rcsb.org/download/{cif_code}.cif")
        return os.path.abspath(local_cif)
    else:
        local_cif = f"AF-{cif_code}-F1-model_v3.cif"
        if not os.path.isfile(local_cif):
            os.system(
                f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{cif_code}-F1-model_v3.cif"
            )
        return os.path.abspath(local_cif)


def get_CA(x):
    xyz = []
    with open(x) as file:
        for line in file:
            line = line.rstrip()
            if line[:4] == "ATOM":
                atom = line[12 : 12 + 4].strip()
                if atom == "CA":
                    resi = line[17 : 17 + 3]
                    resn = int(line[22 : 22 + 5]) - 1
                    x = float(line[30 : 30 + 8])
                    y = float(line[38 : 38 + 8])
                    z = float(line[46 : 46 + 8])
                    xyz.append([x, y, z])
    return np.array(xyz)

def binder_binds_contacts(
    pdb_path, binder_chain, target_chain, contact_residues, cutoff=10.0
):
    """
    Returns True if at least 20% of contact_residues on target_chain
    are contacted by any CA atom of binder_chain within cutoff angstroms.
    Works even if resname is UNK (or non-canonical).
    """
    if isinstance(contact_residues, str):
        if contact_residues.strip() == "":
            return True
        contact_residues = [int(x) for x in contact_residues.split(",") if x.strip()]

    structure = parsePDB(pdb_path)
    if structure is None:
        return False

    def get_chain_ca_atoms_and_resnums(chain_id):
        ca_atoms = []
        ca_resnums = []
        for atom in structure.iterAtoms():
            if atom.getChid() == chain_id and atom.getName() == "CA":
                ca_atoms.append(atom)
                ca_resnums.append(atom.getResnum())
        return ca_atoms, ca_resnums

    binder_ca_atoms, _ = get_chain_ca_atoms_and_resnums(binder_chain)
    target_ca_atoms, target_resnums = get_chain_ca_atoms_and_resnums(target_chain)

    if len(binder_ca_atoms) == 0 or len(target_ca_atoms) == 0:
        return False

    binder_coords = np.array([atom.getCoords() for atom in binder_ca_atoms])
    target_coords = np.array([atom.getCoords() for atom in target_ca_atoms])

    contact_indices = [
        i for i, resnum in enumerate(target_resnums) if resnum in contact_residues
    ]
    if not contact_indices:
        return False

    filtered_target_coords = target_coords[contact_indices]
    filtered_contact_resnums = [target_resnums[i] for i in contact_indices]

    # For each contact residue, check if any binder CA is within cutoff.
    contacted = 0
    for idx, c_resnum in enumerate(filtered_contact_resnums):
        c_coord = filtered_target_coords[idx]
        distances = np.sqrt(np.sum((binder_coords - c_coord) ** 2, axis=-1))
        if np.any(distances < cutoff):
            # print(f"Contacted residue {c_resnum} by binder CA")
            contacted += 1

    if len(filtered_contact_resnums) == 0:
        return False
    return contacted >= 2


def sample_seq(length: int, exclude_P: bool = True, frac_X: float = 0.0) -> str:
    aas = "ACDEFGHIKLMNQRSTVWY" + ("" if exclude_P else "P")
    num_x = round(length * frac_X)
    pool = aas if aas else "X"
    seq_list = ["X"] * num_x + random.choices(pool, k=length - num_x)
    random.shuffle(seq_list)
    return "".join(seq_list)



# Amino acid conversion dict
restype_3to1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M',
}


def extract_sequence_from_structure(pdb_path, chain_id):
    """Extract sequence from PDB file"""
    structure = gemmi.read_structure(str(pdb_path))

    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                seq = []
                for residue in chain:
                    res_name = residue.name.strip().upper()
                    if res_name in restype_3to1:
                        seq.append(restype_3to1[res_name])
                return ''.join(seq)

    raise ValueError(f"Chain {chain_id} not found in {pdb_path}")


def shallow_copy_tensor_dict(d):
    if isinstance(d, dict):
        return {k: shallow_copy_tensor_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [shallow_copy_tensor_dict(x) for x in d]
    if isinstance(d, torch.Tensor):
        return d.detach().clone()
    return d


def smart_split(s):
    if "," in s:
        return [x.strip() for x in s.split(",")]
    if ":" in s:
        return [x.strip() for x in s.split(":")]
    return [x.strip() for x in s.split()] if s else []


def get_boltz_model(
    checkpoint: Optional[str] = None,
    predict_args=None,
    device: Optional[str] = None,
    model_version: str = "boltz2",
    grad_enabled=True,
    no_potentials=True,
) -> Boltz2:
    torch.set_grad_enabled(grad_enabled)
    torch.set_float32_matmul_precision("highest")
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.638  # Default value

    steering_args = BoltzSteeringParams()
    if no_potentials:
        steering_args.fk_steering = False
        steering_args.guidance_update = False

    pairformer_args = (
        PairformerArgsV2() if model_version == "boltz2" else PairformerArgs()
    )
    pairformer_args.v2 = True if model_version == "boltz2" else False
    pairformer_args.activation_checkpointing = True

    msa_args = MSAModuleArgs(
        subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True
    )
    msa_args.activation_checkpointing = True

    model_class = Boltz2 if model_version == "boltz2" else Boltz1
    if model_version == "boltz2":
        model_module = model_class.load_from_checkpoint(
            checkpoint,
            strict=False,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            structure_prediction_training=True,
            no_msa=False,
            no_atom_encoder=False,
            use_templates=True,
            use_templates_v2=True,
            use_trifast=False,
            max_parallel_samples=1,
            steering_args=asdict(steering_args),
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
        )
    elif model_version == "boltz1":
        model_module = Boltz1.load_from_checkpoint(
            checkpoint,
            strict=False,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            structure_prediction_training=True,
            no_msa=False,
            no_atom_encoder=False,
        )
    return model_module


def get_batch(
    target,
    ccd_path,
    ccd_lib,
    max_seqs=0,
    pocket_conditioning=False,
    keep_record=False,
    boltz_model_version=None,
):
    max_seqs = 4096
    structure = target.structure

    coords = np.array([(atom["coords"],) for atom in structure.atoms], dtype=Coords)
    ensemble = np.array([(0, len(coords))], dtype=Ensemble)

    if boltz_model_version == "boltz2":
        structure = StructureV2(
            atoms=structure.atoms,
            bonds=structure.bonds,
            residues=structure.residues,
            chains=structure.chains,
            interfaces=structure.interfaces,
            mask=structure.mask,
            coords=coords,
            ensemble=ensemble,
        )

    elif boltz_model_version == "boltz1":
        structure = Structure(
            atoms=structure.atoms,
            bonds=structure.bonds,
            residues=structure.residues,
            chains=structure.chains,
            interfaces=structure.interfaces,
            mask=structure.mask,
            connections=structure.connections,
        )

    msas = {}
    for chain in target.record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa = np.load(msa_id)
            msas[chain.chain_id] = MSA(**msa)

    input = Input(
        structure,
        msas,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )

    if boltz_model_version == "boltz2":
        tokenizer = Boltz2Tokenizer()
        featurizer = Boltz2Featurizer()
    elif boltz_model_version == "boltz1":
        tokenizer = BoltzTokenizer()
        featurizer = BoltzFeaturizer()

    tokenized = tokenizer.tokenize(input)

    # seed = 42
    # random = np.random.default_rng(seed)
    random = np.random.default_rng()
    if boltz_model_version == "boltz2":
        molecules = {}
        molecules.update(ccd_lib)
        molecules.update(input.extra_mols)
        mol_names = set(tokenized.tokens["res_name"].tolist())
        mol_names = mol_names - set(molecules.keys())
        molecules.update(load_molecules(ccd_path, mol_names))
    options = target.record.inference_options
    if pocket_conditioning:
        pocket_constraints, contact_constraints = (
            options.pocket_constraints,
            options.contact_constraints,
        )
        print("pocket_constraints", pocket_constraints)
        print("contact_constraints", contact_constraints)
        if boltz_model_version == "boltz2":
            batch = featurizer.process(
                tokenized,
                random=random,
                molecules=molecules,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                compute_symmetries=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                inference_contact_constraints=contact_constraints,
                compute_constraint_features=True,
                compute_affinity=False,
            )
        elif boltz_model_version == "boltz1":
            binders, pocket = options.binders, options.pocket
            batch = featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
            )

    else:
        pocket_constraints = None

        if boltz_model_version == "boltz2":
            batch = featurizer.process(
                tokenized,
                random=random,
                molecules=molecules,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                compute_symmetries=False,
                single_sequence_prop=0.0,
                compute_frames=True,
                inference_pocket_constraints=pocket_constraints,
                compute_constraint_features=True,
                compute_affinity=False,
            )
        else:
            batch = featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=None,
                inference_pocket=None,
            )

    if keep_record:
        batch["record"] = target.record

    return batch, structure


def process_msa(chain_id: str, sequence: str, msa_dir: Path) -> bool:
    """Process MSA for a single chain."""
    msa_chain_dir = msa_dir / f"{chain_id}"
    env_dir = msa_chain_dir.with_name(f"{msa_chain_dir.name}_env")
    env_dir.mkdir(exist_ok=True)

    # Run MSA
    unpaired_msa = run_mmseqs2(
        [sequence],
        str(msa_chain_dir),
        use_env=True,
        use_pairing=False,
        host_url="https://api.colabfold.com",
        pairing_strategy="greedy",
    )

    # Save MSA results
    msa_a3m_path = env_dir / "msa.a3m"
    msa_a3m_path.write_text(unpaired_msa[0])

    # Process MSA if not already processed
    msa_npz_path = env_dir / "msa.npz"
    if not msa_npz_path.exists():
        msa = parse_a3m(
            msa_a3m_path,
            taxonomy=None,
            max_seqs=4096,
        )
        msa.dump(msa_npz_path)

    return msa_npz_path


def aggressive_memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()

    for _ in range(3):
        gc.collect()

    torch._dynamo.reset()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    aggressive_memory_cleanup()


def run_prediction(
    data,
    binder_chain,
    seq=None,
    logmd=False,
    name=None,
    ccd_lib=None,
    ccd_path=None,
    boltz_model=None,
    randomly_kill_helix_feature=False,
    negative_helix_constant=0.1,
    device="cpu",
    boltz_model_version="boltz2",
    pocket_conditioning=False,
):
    if seq is not None:
        data["sequences"][chain_to_number[binder_chain]]["protein"]["sequence"] = seq
    target = parse_boltz_schema(
        name,
        data,
        ccd_lib,
        ccd_path,
        boltz_2=True if boltz_model_version == "boltz2" else False,
    )
    batch, structure = get_batch(
        target,
        ccd_path,
        ccd_lib,
        boltz_model_version=boltz_model_version,
        pocket_conditioning=pocket_conditioning,
    )
    batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
    output = boltz_model.predict_step(
        batch,
        batch_idx=0,
        dataloader_idx=0,
        randomly_kill_helix_feature=randomly_kill_helix_feature,
        negative_helix_constant=negative_helix_constant,
        binder_chain=binder_chain,
        logmd=logmd,
        structure=structure,
    )
    return output, structure


def save_pdb(structure, coords, plddts, filename):
    structure.atoms["coords"] = (
        coords[0].detach().cpu().numpy()[: structure.atoms["coords"].shape[0]]
    )
    with open(filename, "w") as f:
        f.write(to_pdb(structure, plddts, boltz2=True))


def design_sequence(
    designer,
    model_type,
    pdb_file,
    chains_to_design="A",
    omit_AA="C",
    bias_AA="",
    temperature=0.02,
    return_logits=False,
):
    seq, logits = designer.run(
        model_type=model_type,
        pdb_path=pdb_file,
        seed=111,
        chains_to_design=chains_to_design,
        bias_AA=bias_AA,
        omit_AA=omit_AA,
        return_logits=return_logits,
        extra_args={
            "--temperature": temperature,
            "--batch_size": 1,
        },
    )
    if return_logits:
        return seq, logits
    return seq[0], logits


def plot_from_pdb(
    pdb_file: str,
    width: int = 400,
    height: int = 400,
    style: str = "cartoon",
    color_by: str = "plddt",
):
    """
    Visualize a structure directly from a PDB file using py3Dmol.

    Args:
        pdb_file (str): Path to the PDB file.
        width (int): Width of the visualization window.
        height (int): Height of the visualization window.
        style (str): Visualization style (default: cartoon).
        color_by (str): Coloring scheme ("none", "plddt", "chain").
    """
    pdb_text = Path(pdb_file).read_text()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")

    # Normalize color_by
    color_by = (color_by or "none").lower()

    if color_by == "plddt":
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {
                style: {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": 50,
                        "max": 90,
                    }
                }
            },
        )
    elif color_by == "chain":
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {style: {"colorscheme": "chain"}},
        )
    else:
        view.setStyle(
            {"and": [{"not": {"resn": ["UNL", "LIG"]}}, {"not": {"hetflag": True}}]},
            {style: {}},
        )

    # Show ligands/heteroatoms as sticks
    view.setStyle(
        {"or": [{"resn": ["UNL", "LIG"]}, {"hetflag": True}]},
        {"stick": {"radius": 0.2}},
    )

    view.zoomTo()
    return view.show()


def plot_run_metrics(
    run_save_dir: str, name: str, run_id: int, num_cycles: int, run_metrics: dict
):
    """Plot per-run figure to run subfolder, given run_metrics as input."""
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    colors = ["#9B59B6", "#E94560", "#FF7F11", "#2ECC71"]
    metrics = [
        (
            "iPTM",
            [
                run_metrics.get(f"cycle_{i}_iptm", float("nan"))
                for i in range(num_cycles + 1)
            ],
            0,
            1,
            "{:.3f}",
        ),
        (
            "pLDDT",
            [
                run_metrics.get(f"cycle_{i}_plddt", float("nan"))
                for i in range(num_cycles + 1)
            ],
            0,
            1,
            "{:.1f}",
        ),
        (
            "iPLDDT",
            [
                run_metrics.get(f"cycle_{i}_iplddt", float("nan"))
                for i in range(num_cycles + 1)
            ],
            0,
            1,
            "{:.1f}",
        ),
        (
            "Alanine Count",
            [
                run_metrics.get(f"cycle_{i}_alanine", float("nan"))
                for i in range(num_cycles + 1)
            ],
            0,
            max(
                [
                    run_metrics.get(f"cycle_{i}_alanine", 0)
                    for i in range(num_cycles + 1)
                ]
            )
            + 2,
            "{}",
        ),
    ]
    design_cycles = list(range(num_cycles + 1))
    for ax, (label, values, ymin, ymax, fmt), color in zip(axs, metrics, colors):
        ax.plot(
            design_cycles,
            values,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=2,
        )
        ax.set(
            xlabel="Design Iteration",
            ylabel=label,
            title=f"{label} (Run {run_id})",
            xticks=design_cycles,
            ylim=(ymin, ymax),
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        for x, y in zip(design_cycles, values):
            ax.annotate(
                fmt.format(y) if not pd.isnull(y) else "",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(f"{run_save_dir}/{name}_run_{run_id}_design_cycle_results.png", dpi=300)
    plt.show()
    # plt.close()