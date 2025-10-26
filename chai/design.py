import os, tempfile, subprocess, glob, json
import numpy as np
import torch
import random
from typing import Dict
from torch import Tensor
import gemmi
import shutil
import random

from chai_lab.chai1 import _bin_centers
class LigandMPNNWrapper:
    def __init__(self, run_py="LigandMPNN/run.py", python="python"):
        self.run_py = run_py
        self.python = python

    def run(self, pdb_path, seed=111, model_type="protein_mpnn",
            temperature=0.1, chains_to_design=None,
            temperature_per_residue=None, extra_args=None, fix_unk=True):
        """
        Run Ligand/ProteinMPNN and return sequences.
        - pdb_path: path to CIF/PDB file
        - seed: random seed
        - model_type: "protein_mpnn", "ligand_mpnn", etc.
        - temperature: global sampling temperature (default 0.1)
        - chains_to_design: string like "B" or "AB"
        - temperature_per_residue: dict like {"A12": 0.5, "A13": 0.2, "B5": 0.8}
        - extra_args: dict of {flag: value}, e.g. {"--batch_size": 8}
        - fix_unk: if True, replace 'UNK' with 'GLY' in a temp copy
        """
        extra_args = dict(extra_args or {})
        with tempfile.TemporaryDirectory() as tmpdir:
            out_folder = tmpdir

            # preprocess CIF/PDB to replace UNK with GLY
            pdb_copy = os.path.join(tmpdir, os.path.basename(pdb_path))
            if fix_unk:
                with open(pdb_path, "r") as fin, open(pdb_copy, "w") as fout:
                    for line in fin:
                        fout.write(line.replace("UNK", "GLY"))
            else:
                # just copy
                with open(pdb_path, "r") as fin, open(pdb_copy, "w") as fout:
                    fout.write(fin.read())

            # handle temperature_per_residue if provided
            temp_json_path = None
            if temperature_per_residue:
                temp_json_path = os.path.join(tmpdir, "temperature_per_residue.json")
                with open(temp_json_path, "w") as f:
                    json.dump(temperature_per_residue, f)

            # build command
            cmd = [
                self.python, self.run_py,
                "--seed", str(seed),
                "--pdb_path", pdb_copy,
                "--out_folder", out_folder,
                "--model_type", model_type,
                "--temperature", str(temperature)
            ]
            if chains_to_design:
                if isinstance(chains_to_design, (list, tuple)):
                    chains_to_design = "".join(chains_to_design)
                cmd += ["--chains_to_design", chains_to_design]

            if temp_json_path:
                cmd += ["--temperature_per_residue", temp_json_path]

            for k, v in extra_args.items():
                cmd += [k, str(v)]

            subprocess.run(" ".join(cmd),
                           shell=True,
                           check=True,
                           stdout=subprocess.DEVNULL,  # Suppress standard output
                           stderr=subprocess.DEVNULL   # Suppress standard error
                          )

            # collect sequences
            fasta_files = glob.glob(os.path.join(out_folder, "seqs", "*.fa"))
            if not fasta_files:
                raise RuntimeError("No FASTA found in output folder.")
            fasta = fasta_files[0]

            seqs = []
            with open(fasta) as f:
                for line in f:
                    if not line.startswith(">"):
                        seqs.append(line.strip())
            return seqs[1:]

def sample_seq(length: int, exclude_P: bool = True, frac_X: float = 0.0) -> str:
  aas = "ACDEFGHIKLMNQRSTVWY" + ("" if exclude_P else "P")
  num_x = round(length * frac_X)
  pool = aas if aas else "X"    
  seq_list = ["X"] * num_x + random.choices(pool, k=length - num_x)
  random.shuffle(seq_list)    
  return "".join(seq_list)

def extract_sequence_from_pdb(pdb_path, chain_id):
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


# Amino acid conversion dict
restype_3to1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M',
}

def extend(a, b, c, L, A, D):
    """
    Place 4th atom given 3 atoms and ideal geometry. Works for single or batched inputs.

    Args:
        a, b, c: Atom positions [3] or [batch, 3]
        L: Bond length from c to new atom
        A: Bond angle at c (radians)
        D: Dihedral angle (radians)

    Returns:
        Position of 4th atom [3] or [batch, 3]
    """
    ba = b - a
    bc = b - c

    x = bc * torch.rsqrt(torch.sum(bc * bc, dim=-1, keepdim=True) + 1e-8)
    z = torch.linalg.cross(ba, x)
    z = z * torch.rsqrt(torch.sum(z * z, dim=-1, keepdim=True) + 1e-8)
    y = torch.linalg.cross(z, x)

    cos_A, sin_A, cos_D, sin_D = np.cos(A), np.sin(A), np.cos(D), np.sin(D)
    L_sin_A = L * sin_A

    return c + L * cos_A * x + L_sin_A * cos_D * y - L_sin_A * sin_D * z

def get_backbone_coords_from_result(result: Dict, batch: Dict) -> tuple[Tensor, Tensor]:
    """
    Extract per-token backbone coordinates (N, CA, C) from prediction.

    Args:
        result: Prediction result from folder.state.result
        batch: Batch dict from folder.state.batch

    Returns:
        backbone_coords: [n_tokens, 3, 3] with N, CA, C coords
        backbone_mask: [n_tokens] bool mask
    """
    inputs = batch["inputs"]

    atom_mask = inputs["atom_exists_mask"].squeeze(0).cpu()
    token_mask = inputs["token_exists_mask"].squeeze(0).cpu()
    n_tokens = token_mask.sum().item()

    # Pad predicted coords
    padded_coords = torch.zeros(atom_mask.shape[0], 3)
    padded_coords[atom_mask] = result['coords']

    # Extract N, CA, C
    bb_indices = inputs["token_backbone_frame_index"].squeeze(0).cpu()[:n_tokens]
    bb_mask = inputs["token_backbone_frame_mask"].squeeze(0).cpu()[:n_tokens]

    return padded_coords[bb_indices], bb_mask


def prepare_refinement_coords(folder, parent_result, parent_batch):
    """
    Prepare coordinates for refinement from parent structure.
    - Combines robust chain-based matching (from Func 1) with
      protein refinement logic (from Func 2).
    - Protein tokens: Copies N, CA, C, O, CB from parent; places other sidechains at parent CB.
    - Ligand tokens: Copied exactly if atom count matches.
    """
    # --- Setup from Function 1 (Robust Data Loading) ---
    parent_coords = parent_result["coords"]
    parent_atom_mask = parent_batch["inputs"]["atom_exists_mask"][0].cpu()
    parent_token_idx = parent_batch["inputs"]["atom_token_index"][0].cpu()
    parent_token_asym_id = parent_batch["inputs"]["token_asym_id"][0].cpu()
    parent_token_entity_type = parent_batch["inputs"]["token_entity_type"][0].cpu()
    
    # Use folder.state.batch or folder.batch depending on your class structure
    new_atom_mask = folder.state.batch["inputs"]["atom_exists_mask"][0].cpu()
    new_token_idx = folder.state.batch["inputs"]["atom_token_index"][0].cpu()
    new_token_asym_id = folder.state.batch["inputs"]["token_asym_id"][0].cpu()
    new_token_entity_type = folder.state.batch["inputs"]["token_entity_type"][0].cpu()

    # --- Added: Setup from Function 2 (Needed for Protein Logic) ---
    parent_atom_within_token = parent_batch["inputs"]["atom_within_token_index"][0].cpu()
    new_atom_within_token = folder.state.batch["inputs"]["atom_within_token_index"][0].cpu()
    parent_bb_indices = parent_batch["inputs"]["token_backbone_frame_index"][0].cpu()

    # --- Initialization ---
    n_atoms_new_padded = new_atom_mask.shape[0]
    new_padded = torch.zeros(n_atoms_new_padded, 3)
    
    parent_padded = torch.zeros(parent_atom_mask.shape[0], 3)
    parent_padded[parent_atom_mask] = parent_coords
    
    # --- Chain Matching Logic (from Function 1) ---
    parent_atom_asym_id = parent_token_asym_id[parent_token_idx]
    new_atom_asym_id = new_token_asym_id[new_token_idx]
    parent_atom_entity_type = parent_token_entity_type[parent_token_idx]
    new_atom_entity_type = new_token_entity_type[new_token_idx]
    
    parent_asym_ids = torch.unique(parent_atom_asym_id[parent_atom_mask])
    new_asym_ids = torch.unique(new_atom_asym_id[new_atom_mask])
    common_asym_ids = set(parent_asym_ids.tolist()) & set(new_asym_ids.tolist())
    
    for asym_id in sorted(common_asym_ids):
        parent_chain_mask = (parent_atom_asym_id == asym_id) & parent_atom_mask
        new_chain_mask = (new_atom_asym_id == asym_id) & new_atom_mask
        
        # Get entity type (assuming all atoms in chain have same type)
        # Add check for empty mask to prevent indexing error
        if not parent_chain_mask.any() or not new_chain_mask.any():
            continue
            
        parent_entity = parent_atom_entity_type[parent_chain_mask][0].item()
        new_entity = new_atom_entity_type[new_chain_mask][0].item()
        
        assert parent_entity == new_entity, f"Chain {asym_id} changed entity type!"
        
        # --- Ligand Logic (from Function 1) ---
        if parent_entity == EntityType.LIGAND.value:
            parent_indices = torch.where(parent_chain_mask)[0]
            new_indices = torch.where(new_chain_mask)[0]
            
            assert len(parent_indices) == len(new_indices), \
                f"Ligand atom count mismatch! Parent: {len(parent_indices)}, New: {len(new_indices)}"
            
            new_padded[new_indices] = parent_padded[parent_indices]
            
        elif parent_entity == EntityType.PROTEIN.value:
            # Find common tokens *within this chain*
            parent_tokens_in_chain = parent_token_idx[parent_chain_mask]
            new_tokens_in_chain = new_token_idx[new_chain_mask]
            
            common_tokens = set(torch.unique(parent_tokens_in_chain).tolist()) & set(torch.unique(new_tokens_in_chain).tolist())
            
            for token_id in common_tokens:
                # Get masks for this specific token
                parent_token_atoms_mask = (parent_token_idx == token_id) & parent_atom_mask
                new_token_atoms_mask = (new_token_idx == token_id) & new_atom_mask
                
                # 1. Find/calculate parent CB position
                parent_CB_atoms = parent_token_atoms_mask & (parent_atom_within_token == 3) # 3 = CB
                if parent_CB_atoms.any():
                    CB_pos = parent_padded[parent_CB_atoms][0]
                else:
                    # Glycine case: calculate CB (assuming 'extend' is defined)
                    N_idx, CA_idx, C_idx = parent_bb_indices[token_id]
                    N, CA, C = parent_padded[N_idx], parent_padded[CA_idx], parent_padded[C_idx]
                    CB_pos = extend(C, N, CA, 1.522, 1.927, -2.143) # Magic numbers from F2

                # 2. Set all new token atoms to this CB position
                new_padded[new_token_atoms_mask] = CB_pos

                # 3. Overwrite backbone atoms (N, CA, C, O, CB)
                for atom_idx in [0, 1, 2, 3, 4]:
                    parent_atoms = parent_token_atoms_mask & (parent_atom_within_token == atom_idx)
                    new_atoms = new_token_atoms_mask & (new_atom_within_token == atom_idx)
                    
                    if parent_atoms.any() and new_atoms.any():
                        # Ensure we're only taking the first (and only) atom
                        new_padded[new_atoms] = parent_padded[parent_atoms][0] 
                
    return new_padded[new_atom_mask]

def is_smiles(seq):
    """Detect if sequence is SMILES string vs protein sequence"""
    smiles_chars = set('()[]=#@+-0123456789')
    return bool(set(seq.upper()) & smiles_chars)

def extract_sequence_from_pdb(pdb_path, chain_id):
    """Extract amino acid sequence from PDB file for a given chain"""
    structure = gemmi.read_structure(str(pdb_path))
    
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                subchains = list(chain.subchains())
                if subchains:
                    residues = list(subchains[0].first_conformer())
                    # Convert 3-letter codes to 1-letter
                    from chai_lab.data.residue_constants import restype_3to1
                    sequence = ''.join([restype_3to1.get(res.name, 'X') for res in residues])
                    return sequence
    
    raise ValueError(f"Chain {chain_id} not found in {pdb_path}")

def kabsch_rotation_matrix(
    mobile_centered: Tensor,
    target_centered: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute optimal rotation matrix using Kabsch algorithm.
    
    Args:
        mobile_centered: [N, 3] centered coordinates to rotate
        target_centered: [N, 3] centered reference coordinates
        weights: Optional [N, 1] weights for each point
        
    Returns:
        [3, 3] rotation matrix
    """
    # Compute covariance matrix
    if weights is not None:
        H = mobile_centered.T @ (weights * target_centered)
    else:
        H = mobile_centered.T @ target_centered
    
    # SVD
    U, S, Vt = torch.linalg.svd(H)
    R = U @ Vt
    
    # Handle reflection case
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    
    return R

def weighted_kabsch_align(
    mobile: Tensor,
    target: Tensor,
    weights: Optional[Tensor] = None,
    mobile_mask: Optional[Tensor] = None,
    target_mask: Optional[Tensor] = None,
) -> Tensor:

    # Handle masks - use subset for alignment computation
    if mobile_mask is not None and target_mask is not None:
        mobile_sub = mobile[mobile_mask]
        target_sub = target[target_mask]
        weights_sub = weights[mobile_mask] if weights is not None else None
    else:
        mobile_sub = mobile
        target_sub = target
        weights_sub = weights
    
    # Expand weights if needed
    if weights_sub is not None and weights_sub.ndim == 1:
        weights_sub = weights_sub.unsqueeze(-1)
    
    # Compute weighted centroids from subset
    if weights_sub is not None:
        weights_sum = weights_sub.sum(dim=0, keepdim=True).clamp(min=1e-8)
        centroid_mobile = (mobile_sub * weights_sub).sum(dim=0, keepdim=True) / weights_sum
        centroid_target = (target_sub * weights_sub).sum(dim=0, keepdim=True) / weights_sum
    else:
        centroid_mobile = mobile_sub.mean(dim=0, keepdim=True)
        centroid_target = target_sub.mean(dim=0, keepdim=True)
    
    # Center subset coordinates
    mobile_sub_centered = mobile_sub - centroid_mobile
    target_sub_centered = target_sub - centroid_target
    
    # Compute rotation matrix from subset
    R = kabsch_rotation_matrix(mobile_sub_centered, target_sub_centered, weights_sub)
    
    # Apply transformation to ALL mobile coordinates
    mobile_centered = mobile - centroid_mobile
    mobile_aligned = mobile_centered @ R + centroid_target
    
    return mobile_aligned

def compute_rmsd(coords1: Tensor, coords2: Tensor, mask: Optional[Tensor] = None) -> float:

    if mask is not None:
        coords1 = coords1[mask]
        coords2 = coords2[mask]
    
    return torch.sqrt(torch.mean((coords1 - coords2) ** 2)).item()

def compute_ca_rmsd(
    coords1: Tensor,
    coords2: Tensor,
    mode: str = "all",
    n_target: Optional[int] = None,
) -> float:
    """
    Compute CA/reference atom RMSD with different alignment modes.
    
    Args:
        coords1: [N, 3, 3] backbone coordinates (N-CA-C)
        coords2: [N, 3, 3] backbone coordinates
        mode: Alignment mode:
            - "all": Align all atoms, measure all RMSD
            - "target_align_binder_rmsd": Align by target, measure binder RMSD
            - "binder_align_ligand_com_rmsd": Align by binder, measure ligand COM distance
        n_target: Number of target residues (required for multi-chain modes)
        
    Returns:
        RMSD or distance value as float
    """
    # Extract CA atoms (middle atom)
    coords1 = coords1[:, 1, :].float()
    coords2 = coords2[:, 1, :].float()
    
    if mode == "all":
        # Simple case: align everything, measure everything
        coords2_aligned = weighted_kabsch_align(coords2, coords1)
        return compute_rmsd(coords1, coords2_aligned)
    
    elif mode == "target_align_binder_rmsd":
        assert n_target is not None, "n_target required for target_align_binder_rmsd mode"
        
        # Create mask for target region
        n_total = coords1.shape[0]
        target_mask = torch.zeros(n_total, dtype=bool, device=coords1.device)
        target_mask[:n_target] = True
        
        # Align coords2 to coords1 using ONLY target region
        coords2_aligned = weighted_kabsch_align(
            coords2, coords1,
            mobile_mask=target_mask,
            target_mask=target_mask,
        )
        
        # Extract binder regions and measure RMSD
        binder_mask = ~target_mask
        return compute_rmsd(coords1, coords2_aligned, mask=binder_mask)
    
    elif mode == "binder_align_ligand_com_rmsd":
        assert n_target is not None, "n_target required for binder_align_ligand_com_rmsd mode"
        
        # Split into ligand and binder
        ligand1, binder1 = coords1[:n_target], coords1[n_target:]
        ligand2, binder2 = coords2[:n_target], coords2[n_target:]
        
        # Compute ligand centers of mass
        ligand_com1 = ligand1.mean(dim=0)
        ligand_com2 = ligand2.mean(dim=0)
        
        # Align binder2 to binder1
        binder2_aligned = weighted_kabsch_align(binder2, binder1)
        
        # Get transformation parameters from binder alignment
        centroid_binder1 = binder1.mean(dim=0)
        centroid_binder2 = binder2.mean(dim=0)
        binder1_centered = binder1 - centroid_binder1
        binder2_centered = binder2 - centroid_binder2
        R = kabsch_rotation_matrix(binder2_centered, binder1_centered)
        
        # Apply same transformation to ligand COM
        ligand_com2_aligned = (ligand_com2 - centroid_binder2) @ R + centroid_binder1
        
        # Return distance between ligand COMs
        return torch.sqrt(torch.sum((ligand_com1 - ligand_com2_aligned) ** 2)).item()
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def optimize_protein_design(
    folder,
    designer,
    initial_seq,
    target_seq=None,
    target_pdb=None,
    target_chain=None,
    prefix="test",
    n_steps=5,
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
    num_diffn_samples=1,
    temperature=0.1,
    use_esm=False,
    use_esm_target=False,
    use_alignment=True,
    align_to="all",
    scale_temp_by_plddt=False,
    partial_diffusion=0.0,
    pde_cutoff_intra=1.5,
    pde_cutoff_inter=3.0,
    omit_AA=None,
    randomize_template_sequence=True,
    cyclic=False,
    final_validation=True,
    verbose=False,
):
    """
    Optimize protein design through iterative folding and sequence design.
    
    Args:
        folder: ChaiFolder instance
        designer: LigandMPNNWrapper instance
        initial_seq: Initial sequence to optimize
        target_seq: Target (protein sequence OR SMILES for ligand). If None with target_pdb, extracts from PDB.
        target_pdb: Optional template PDB
        target_chain: Chain ID in template PDB (required if target_pdb provided)
        prefix: Output file prefix
        n_steps: Number of optimization steps
        num_trunk_recycles: Trunk recycles per fold
        num_diffn_timesteps: Diffusion timesteps
        num_diffn_samples: Number of Diffusion samples per step
        temperature: MPNN sampling temperature
        use_esm: Use ESM for binder/unconditional (all iterations)
        use_esm_target: Use ESM for target protein (all iterations, ignored for ligands)
        use_alignment: Enable alignment during diffusion
        scale_temp_by_plddt: Scale MPNN temperature by inverse pLDDT
        partial_diffusion: Use partial diffusion refinement
        pde_cutoff_intra: Intra-chain PDE cutoff (Å)
        pde_cutoff_inter: Inter-chain PDE cutoff (Å)
        omit_AA: Amino acids to exclude from MPNN
        randomize_template_sequence: Randomize template sequence identity
        cyclic: Enable cyclic topology
        final_validation: Run final fold without templates
        verbose: Print detailed progress
    
    Supports:
        - Unconditional design (no target)
        - Protein-protein binder design
        - Protein-ligand binder design (target_seq = SMILES)
    """
    
    # Extract target sequence from PDB if not provided
    if target_pdb is not None and target_seq is None:
        if target_chain is None:
            raise ValueError("target_chain must be specified when using target_pdb")
        target_seq = extract_sequence_from_pdb(target_pdb, target_chain)
        if verbose:
            print(f"{prefix} | Extracted target sequence from {target_pdb} chain {target_chain}: {target_seq[:60]}...")
    
    # Detect target type
    is_ligand_target = False
    if target_seq is not None:
        is_ligand_target = is_smiles(target_seq)
    
    is_binder_design = target_seq is not None
    mpnn_model_type = "ligand_mpnn" if is_ligand_target else "soluble_mpnn"
    target_entity_type = "ligand" if is_ligand_target else "protein"
    
    # Calculate PDE bins
    bin_centers = _bin_centers(0.0, 32.0, 64)
    pde_bins_intra = (bin_centers <= pde_cutoff_intra).sum().item()
    pde_bins_inter = (bin_centers <= pde_cutoff_inter).sum().item()
    
    def compute_pae_metrics(pae, n_target):
        """Compute PAE and iPAE"""
        if not is_binder_design:
            return {'pae': pae.mean().item(), 'ipae': None}
        
        mean_pae = pae.mean().item()
        if is_ligand_target:
            target_to_binder = pae[:n_target, n_target:].min(1).values.mean().item()
            binder_to_target = pae[n_target:, :n_target].min(0).values.mean().item()
        else:
            target_to_binder = pae[:n_target, n_target:].mean().item()
            binder_to_target = pae[n_target:, :n_target].mean().item()
            
        ipae = (target_to_binder + binder_to_target) / 2        
        return {'pae': mean_pae, 'ipae': ipae}
    
    def compute_template_weight(prev_pde, n_target):
        """Compute PDE-based template weight"""
        if not is_binder_design:
            weight = prev_pde[..., :pde_bins_intra].sum(-1)
        
        else:
            n_total = prev_pde.shape[0]
            weight = torch.ones(n_total, n_total)            
            weight[n_target:, n_target:] = prev_pde[n_target:, n_target:, :pde_bins_intra].sum(-1)
            weight[:n_target, n_target:] = prev_pde[:n_target, n_target:, :pde_bins_inter].sum(-1)
            weight[n_target:, :n_target] = prev_pde[n_target:, :n_target, :pde_bins_inter].sum(-1)
        
        return weight
    
    def fold_sequence(seq, is_first_iteration, prev_pdb=None, prev_pde=None, prev_n_target=None, prev_result=None, prev_batch=None):
        """Fold sequence and return metrics"""
        chains = []
        
        if is_binder_design:
            # Target chain
            align_target_weight = 10.0 if align_to in ["target","ligand"] else 1.0                
            if is_first_iteration:
                if target_pdb is not None and not is_ligand_target:
                    # Protein target with template
                    target_opts = {
                        "use_esm": use_esm_target,
                        "template_pdb": target_pdb,
                        "template_chain_id": target_chain,
                        "align":align_target_weight,
                    }
                else:
                    # Protein target without template OR ligand target
                    target_opts = {"use_esm": use_esm_target and not is_ligand_target}
            else:
                if is_ligand_target:
                    # Ligand: no ESM
                    target_opts = {
                        "use_esm": False,
                        "align":align_target_weight,
                    }
                else:
                    # Protein target: use template + optional ESM
                    target_opts = {
                        "use_esm": use_esm_target,
                        "template_pdb": prev_pdb,
                        "template_chain_id": "A",
                        "randomize_template_sequence": False,
                        "align":align_target_weight,
                    }
            chains.append([target_seq, "A", target_entity_type, target_opts])
            
            # Binder chain
            align_binder_weight = 10.0 if align_to == "binder" else 1.0                
            binder_opts = {"use_esm": use_esm, "cyclic": cyclic, "align":align_binder_weight}
            if not is_first_iteration:
                binder_opts.update({
                    "template_pdb": prev_pdb,
                    "template_chain_id": "B",
                    "randomize_template_sequence": randomize_template_sequence,                    
                })
            chains.append([seq, "B", "protein", binder_opts])
        else:
            # Unconditional
            opts = {"use_esm": use_esm, "cyclic": cyclic}
            if not is_first_iteration:
                opts.update({
                    "template_pdb": prev_pdb,
                    "template_chain_id": "A",
                    "randomize_template_sequence": randomize_template_sequence,
                })
            chains.append([seq, "A", "protein", opts])
        
        # Fold
        template_weight = None if is_first_iteration else compute_template_weight(prev_pde, prev_n_target)
        
        folder.prep_inputs(chains)
        
        # Compute n_target from actual structure
        token_exists = folder.state.batch["inputs"]["token_exists_mask"][0].cpu()
        n_total_tokens = token_exists.sum().item()
        n_target_tokens = (n_total_tokens - len(seq)) if is_binder_design else None
        
        folder.get_embeddings()
        folder.run_trunk(num_trunk_recycles=num_trunk_recycles, template_weight=template_weight)
        
        # Partial diffusion
        refine_coords = None
        refine_step = None
        if partial_diffusion and prev_result is not None:
            refine_coords = prepare_refinement_coords(folder, prev_result, prev_batch)
            refine_step = int(num_diffn_timesteps * partial_diffusion)

        # Sample
        folder.sample(
            num_diffn_timesteps=num_diffn_timesteps,
            num_diffn_samples=num_diffn_samples,
            use_alignment=use_alignment,
            refine_from_coords=refine_coords,
            refine_from_step=refine_step
        )    

        result = folder.state.result
        pae_metrics = compute_pae_metrics(result["pae"], n_target_tokens if n_target_tokens else 0)
        backbone_coords, _ = get_backbone_coords_from_result(result, folder.state.batch)
        return {
            'plddt': result["plddt"].numpy().mean(),
            'ptm': result["ptm"].item(),
            'iptm': result["iptm"].item() if is_binder_design else 0.0,
            'pae': pae_metrics['pae'],
            'ipae': pae_metrics['ipae'],
            'ranking_score': result["ranking_score"],
            'backbone': backbone_coords,
            'pde': result["pde"],
            'sequence': seq,
            'n_target': n_target_tokens,
        }
    
    def design_sequence(prev_pdb, step, prev_plddt=None):
        """Design sequence with MPNN"""
        temp_per_residue = None
        if scale_temp_by_plddt and prev_plddt is not None:
            chain = "B" if is_binder_design else "A"
            inv_plddt = np.square(1 - prev_plddt)
            temp_per_residue = {f"{chain}{i+1}": float(v) for i, v in enumerate(inv_plddt)}
        
        extra_args = {"--batch_size": 1}
        if mpnn_model_type == "ligand_mpnn":
            extra_args["--checkpoint_ligand_mpnn"] = "./model_params/ligandmpnn_v_32_020_25.pt"
        else:
            extra_args["--checkpoint_soluble_mpnn"] = "./model_params/solublempnn_v_48_020.pt"
        if omit_AA:
            extra_args["--omit_AA"] = omit_AA
        
        sequences = designer.run(
            model_type=mpnn_model_type,
            pdb_path=prev_pdb,
            seed=111 + step,
            chains_to_design="B" if is_binder_design else "A",
            temperature=temperature,
            temperature_per_residue=temp_per_residue,
            extra_args=extra_args,
        )
        
        seq = sequences[0]
        if is_binder_design:
            seq = seq.split(":")[-1]
        
        return seq
    
    def format_metrics(out, rmsd=None):
        """Format metrics for printing"""
        msg = f"score={out['ranking_score']:.3f} plddt={out['plddt']:.1f} ptm={out['ptm']:.3f}"
        if is_binder_design:
            msg += f" iptm={out['iptm']:.3f} ipae={out['ipae']:.2f}"
        else:
            msg += f" pae={out['pae']:.2f}"
        if rmsd is not None:
            msg += f" rmsd={rmsd:.2f}"
        return msg
    
    # === MAIN LOOP ===
    if verbose and is_binder_design:
        mode = f"{'Ligand' if is_ligand_target else 'Protein'}-binder design"
        print(f"{prefix} | Mode: {mode}")
        if use_esm:
            print(f"{prefix} | Using ESM for binder (all iterations)")
        if use_esm_target and not is_ligand_target:
            print(f"{prefix} | Using ESM for target (all iterations)")
    
    # Step 0
    prev_out = fold_sequence(initial_seq, is_first_iteration=True)
    prev_pdb = f"{prefix}_s0.cif"
    folder.save(prev_pdb)
    
    best_state = folder.save_state()
    best_out = prev_out
    best_step = 0
    
    print(f"{prefix} | Step 0: {format_metrics(prev_out)}")
    
    prev_result = folder.state.result
    prev_batch = folder.state.batch
    
    # Optimization steps
    for step in range(n_steps):
        prev_plddt = None
        if scale_temp_by_plddt:
            prev_plddt = folder.state.result["plddt"].numpy()
            if is_binder_design:
                n_binder = len(initial_seq)
                prev_plddt = prev_plddt[-n_binder:]
        
        new_seq = design_sequence(prev_pdb, step, prev_plddt)
        
        if verbose:
            print(f"{prefix} | Designed: {new_seq[:60]}{'...' if len(new_seq) > 60 else ''}")
        
        out = fold_sequence(
            new_seq, 
            is_first_iteration=False, 
            prev_pdb=prev_pdb, 
            prev_pde=prev_out["pde"],
            prev_n_target=prev_out['n_target'],
            prev_result=prev_result, 
            prev_batch=prev_batch
        )
        
        # Compute RMSD
        if is_binder_design:
            if is_ligand_target:
                rmsd = compute_ca_rmsd(
                    prev_out["backbone"], out["backbone"],
                    mode="binder_align_ligand_com_rmsd",
                    n_target=prev_out['n_target']
                )
            else:
                rmsd = compute_ca_rmsd(
                    prev_out["backbone"], out["backbone"],
                    mode="target_align_binder_rmsd",
                    n_target=prev_out['n_target']
                )
        else:
            rmsd = compute_ca_rmsd(prev_out["backbone"], out["backbone"], mode="all")
        
        print(f"{prefix} | Step {step+1}: {format_metrics(out, rmsd)}")
        
        if out['ranking_score'] > best_out['ranking_score']:
            best_state = folder.save_state()
            best_out = out
            best_step = step + 1
        
        prev_out = out
        prev_result = folder.state.result
        prev_batch = folder.state.batch
        prev_pdb = f"{prefix}_s{step+1}.cif"
        folder.save(prev_pdb)
    
    # Restore best
    folder.restore_state(best_state)
    folder.save(f"{prefix}_best.cif")
    print(f"{prefix} | Best step: {best_step} ({format_metrics(best_out)})")
    
    # Validation
    if final_validation and n_steps > 0:
        print(f"{prefix} | Running final validation (original target, no binder template)...")
        
        chains = []
        if is_binder_design:
            # Target: use original context
            if is_ligand_target:
                target_opts = {"use_esm": False}
            elif target_pdb is not None:
                target_opts = {
                    "use_esm": use_esm_target,
                    "template_pdb": target_pdb,
                    "template_chain_id": target_chain,
                }
            else:
                target_opts = {"use_esm": use_esm_target}
            chains.append([target_seq, "A", target_entity_type, target_opts])
            
            # Binder: no template, but can use ESM
            binder_opts = {"use_esm": use_esm, "cyclic": cyclic}
            chains.append([best_out['sequence'], "B", "protein", binder_opts])
        else:
            # Unconditional: no template, but can use ESM
            opts = {"use_esm": use_esm, "cyclic": cyclic}
            chains.append([best_out['sequence'], "A", "protein", opts])
        
        folder.prep_inputs(chains)
        folder.get_embeddings()
        folder.run_trunk(num_trunk_recycles=num_trunk_recycles, template_weight=None)
        folder.sample(num_diffn_timesteps=num_diffn_timesteps,
                      num_diffn_samples=num_diffn_samples,
                      use_alignment=use_alignment)
        
        val_result = folder.state.result
        val_backbone, _ = get_backbone_coords_from_result(val_result, folder.state.batch)
        val_pae = compute_pae_metrics(val_result["pae"], best_out['n_target'] if best_out['n_target'] else 0)
        
        validation_out = {
            'plddt': val_result["plddt"].numpy().mean(),
            'ptm': val_result["ptm"].item(),
            'iptm': val_result["iptm"].item() if is_binder_design else 0.0,
            'pae': val_pae['pae'],
            'ipae': val_pae['ipae'],
            'ranking_score': val_result["ranking_score"],
            'backbone': val_backbone,
            'sequence': best_out['sequence'],
        }
        
        # Validation RMSD
        if is_binder_design:
            if is_ligand_target:
                val_rmsd = compute_ca_rmsd(
                    best_out["backbone"], validation_out["backbone"],
                    mode="binder_align_ligand_com_rmsd",
                    n_target=best_out['n_target']
                )
            else:
                val_rmsd = compute_ca_rmsd(
                    best_out["backbone"], validation_out["backbone"],
                    mode="target_align_binder_rmsd",
                    n_target=best_out['n_target']
                )
        else:
            val_rmsd = compute_ca_rmsd(best_out["backbone"], validation_out["backbone"], mode="all")
        
        print(f"{prefix} | Validation: {format_metrics(validation_out, val_rmsd)}")
        folder.save(f"{prefix}_validation.cif")
        best_out['validation'] = validation_out
    
    return best_out
