import subprocess

# protein binding design
cmd = [
    "python", "boltz/design.py",
    "--num_designs", "3",
    "--num_cycles", "7",
    "--protein_seqs", "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE",
    "--protein_ids", "B",
    "--protein_msas", "",
    "--gpu_id", "2",
    "--name", "PDL1_mix_aa",
    "--min_design_protein_length", "90",
    "--max_design_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot"
]
subprocess.run(cmd, check=True)

#multimer binding design
cmd = [
    "python", "boltz/design.py",
    "--num_designs", "3",
    "--num_cycles", "7",
    "--protein_seqs", "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE:AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE",
    "--protein_ids", "B:C",
    "--protein_msas", "",
    "--gpu_id", "2",
    "--name", "PDL1_double_mix_aa",
    "--min_design_protein_length", "90",
    "--max_design_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot"
]
subprocess.run(cmd, check=True)


#protein binder with contact residues
cmd = [
    "python", "boltz/design.py",
    "--num_designs", "3",
    "--num_cycles", "7",
    "--protein_seqs", "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE",
    "--protein_ids", "B",
    "--protein_msas", "",
    "--gpu_id", "2",
    "--contact_residues", "29,277,279,293,294,295,318,319,320,371",
    "--add_constraints",
    "--name", "PDL1_contact_residues_mix_aa",
    "--min_design_protein_length", "90",
    "--max_design_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot"
]
subprocess.run(cmd, check=True)

#protein + small molecule binding design
cmd = [
    "python", "boltz/design.py",
    "--num_designs", "3",
    "--num_cycles", "7",
    "--protein_seqs", "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE",
    "--protein_ids", "B",
    "--protein_msas", "",
    "--gpu_id", "2",
    "--name", "PDL1_SAM_mix_aa",
    "--ligand_ccd", "SAM",
    "--ligand_id", "C",
    "--min_design_protein_length", "90",
    "--max_design_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot"
]
subprocess.run(cmd, check=True)


#nucleic acid binding design
cmd = [
    "python", "boltz/design.py",
    "--num_designs", "3",
    "--num_cycles", "7",
    "--gpu_id", "2",
    "--name", "rna_mix_aa",
    "--nucleic_seq", "AGAGAGA",
    "--nucleic_type", "rna",
    "--nucleic_id", "C",
    "--min_design_protein_length", "90",
    "--max_design_protein_length", "150",
    "--high_iptm_threshold", "0.7",
    "--use_msa_for_af3",
    "--plot"
]
subprocess.run(cmd, check=True)