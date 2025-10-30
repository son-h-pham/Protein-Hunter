# af3_generate_rna_msa is implemented from ColabNuFold (https://colab.research.google.com/github/kiharalab/nufold/blob/master/ColabNuFold.ipynb#scrollTo=KDs4o5Bv35MI)

import json
import logging
import os
import subprocess
import tempfile
import time
import urllib
from pathlib import Path

import yaml
from tqdm import tqdm

from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m

rna_default_setting = {
    "use_rfam_db": True,
    "use_rnacentral_db": True,
    "use_ntrna_db": False,
    "max_sequences_per_db": 10000,
    "e_value": 0.001,
    "time_limit_minutes": 120,
    "n_cpu": 2,  # Default to 2 cores for Colab
}

af3_database_settings = {
    "Rfam": rna_default_setting["use_rfam_db"],
    "RNAcentral": rna_default_setting["use_rnacentral_db"],
    "NT_RNA": rna_default_setting["use_ntrna_db"],
    "time_limit_minutes": rna_default_setting["time_limit_minutes"],
    "max_sequences_per_db": rna_default_setting["max_sequences_per_db"],
    "e_value": rna_default_setting["e_value"],
    "n_cpu": rna_default_setting["n_cpu"],  # Default to 2 cores for Colab
}


RNA_CHAIN = "polyribonucleotide"
SHORT_SEQUENCE_CUTOFF = 50
SOURCE = "https://storage.googleapis.com/alphafold-databases/v3.0"
RNA_DATABASE_INFO = {
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta": "RNAcentral",
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta": "NT_RNA",
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta": "Rfam",
}

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
}


def process_msa(chain_id: str, sequence: str, pdb_code: str, msa_dir: Path) -> bool:
    """Process MSA for a single chain."""
    msa_chain_dir = msa_dir / f"{pdb_code}_{chain_id}"
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

    logger.info(f"Processed MSA for {pdb_code} chain {chain_id}")
    return msa_a3m_path


def download_with_progress(url, dest_path):
    """Download a file with a progress bar"""
    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.info().get("Content-Length", 0))
            desc = f"Downloading {os.path.basename(dest_path)}"
            with tqdm(total=file_size, unit="B", unit_scale=True, desc=desc) as pbar:
                with open(dest_path, "wb") as out_file:
                    while True:
                        buffer = response.read(8192)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        pbar.update(len(buffer))
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def download_selected_databases(database_settings, afdb_dir):
    """Download only the databases selected in the settings"""
    selected_dbs = []
    if database_settings.get("Rfam", False):
        selected_dbs.append("Rfam")
    if database_settings.get("RNAcentral", False):
        selected_dbs.append("RNAcentral")
    if database_settings.get("NT_RNA", False):
        selected_dbs.append("NT_RNA")

    if not selected_dbs:
        print("⚠️ No databases selected for download!")
        return

    # Convert selected_dbs to appropriate file names
    selected_db_files = []
    for db_file, db_key in RNA_DATABASE_INFO.items():
        if db_key in selected_dbs:
            selected_db_files.append(db_file)

    if not selected_db_files:
        print("⚠️ No databases selected for download!")
        return

    print(
        f"🌐 Downloading {len(selected_db_files)} RNA databases: {', '.join([RNA_DATABASE_INFO[db] for db in selected_db_files])}"
    )

    # Check if we already have the databases
    missing_dbs = []
    for db in selected_db_files:
        db_path = os.path.join(afdb_dir, db)
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            missing_dbs.append(db)

    if not missing_dbs:
        print("✅ All selected databases already downloaded.")
        return

    # Create progress bar for overall process
    with tqdm(
        total=len(missing_dbs), desc="Overall progress", unit="db", position=0
    ) as main_pbar:
        # Download and decompress each database
        for db in missing_dbs:
            dest_path = os.path.join(afdb_dir, f"{db}.zst")
            final_path = os.path.join(afdb_dir, db)

            if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                print(f"✅ {db} already exists, skipping.")
                main_pbar.update(1)
                continue

            # Download the compressed file
            print(f"📥 Downloading {db} ({RNA_DATABASE_INFO[db]})...")
            url = f"{SOURCE}/{db}.zst"
            if download_with_progress(url, dest_path):
                # Decompress with zstd
                print(f"📦 Decompressing {db}...")
                try:
                    subprocess.run(
                        ["zstd", "--decompress", "-f", dest_path, "-o", final_path],
                        check=True,
                    )
                    print(f"✅ Successfully processed {db}")

                    # Remove the compressed file
                    os.remove(dest_path)
                except Exception as e:
                    print(f"❌ Error decompressing {db}: {e}")

            main_pbar.update(1)

    # Final check to see what we have
    existing_dbs = []
    for db in selected_db_files:
        db_path = os.path.join(afdb_dir, db)
        if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            existing_dbs.append(db)

    if len(existing_dbs) == len(selected_db_files):
        print("🎉 All selected databases downloaded and ready for use.")
    else:
        print(f"⚠️ Downloaded {len(existing_dbs)}/{len(selected_db_files)} databases.")
        print(
            f"✅ Available: {', '.join([RNA_DATABASE_INFO[db] for db in existing_dbs])}"
        )
        missing = [db for db in selected_db_files if db not in existing_dbs]
        print(f"❌ Missing: {', '.join([RNA_DATABASE_INFO[db] for db in missing])}")


def create_query_fasta_file(sequence, path, linewidth=80):
    """Creates a fasta file with the sequence"""
    with open(path, "w") as f:
        f.write(">query\n")
        i = 0
        while i < len(sequence):
            f.write(f"{sequence[i : (i + linewidth)]}\n")
            i += linewidth


def run_command(cmd, cmd_name):
    """Run a command and handle errors"""
    logging.info(f"Running {cmd_name}: {cmd}")
    start_time = time.time()
    try:
        completed_process = subprocess.run(
            cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
    except subprocess.CalledProcessError as e:
        logging.exception(f"{cmd_name} failed.\nstdout: {e.stdout}\nstderr: {e.stderr}")
        raise RuntimeError(
            f"{cmd_name} failed\nstdout: {e.stdout}\nstderr: {e.stderr}"
        ) from e

    end_time = time.time()
    logging.info(f"Finished {cmd_name} in {end_time - start_time:.3f} seconds")
    return completed_process


def parse_fasta(fasta_string):
    """Parse a FASTA string into sequences and descriptions"""
    sequences = []
    descriptions = []

    lines = fasta_string.strip().split("\n")
    current_seq = ""
    current_desc = ""

    for line in lines:
        if line.startswith(">"):
            if current_seq:  # Save the previous sequence
                sequences.append(current_seq)
                descriptions.append(current_desc)
            current_desc = line[1:].strip()  # Remove the '>' character
            current_seq = ""
        else:
            current_seq += line.strip()

    # Add the last sequence
    if current_seq:
        sequences.append(current_seq)
        descriptions.append(current_desc)

    return sequences, descriptions


def convert_stockholm_to_a3m(stockholm_path, max_sequences=None):
    """Convert Stockholm format MSA to A3M format"""
    with open(stockholm_path) as stockholm_file:
        descriptions = {}
        sequences = {}
        reached_max_sequences = False

        # First pass: extract sequences
        for line in stockholm_file:
            reached_max_sequences = max_sequences and len(sequences) >= max_sequences
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            seqname, aligned_seq = line.split(maxsplit=1)
            if seqname not in sequences:
                if reached_max_sequences:
                    continue
                sequences[seqname] = ""
            sequences[seqname] += aligned_seq

        if not sequences:
            return ""

        # Second pass: extract descriptions
        stockholm_file.seek(0)
        for line in stockholm_file:
            line = line.strip()
            if line[:4] == "#=GS":
                columns = line.split(maxsplit=3)
                seqname, feature = columns[1:3]
                value = columns[3] if len(columns) == 4 else ""
                if feature != "DE":
                    continue
                if reached_max_sequences and seqname not in sequences:
                    continue
                descriptions[seqname] = value
                if len(descriptions) == len(sequences):
                    break

    # Convert Stockholm to A3M
    a3m_sequences = {}
    query_sequence = next(iter(sequences.values()))
    for seqname, sto_sequence in sequences.items():
        # Align sequence to gapless query (simplified version)
        a3m_seq = ""
        query_idx = 0
        for i, char in enumerate(sto_sequence):
            if query_sequence[i] == "-":
                if char != "-":
                    a3m_seq += char.lower()  # Add as lowercase (insertion)
            else:  # Query has a residue here
                a3m_seq += char  # Add as is (match/mismatch/deletion)
                query_idx += 1
        a3m_sequences[seqname] = a3m_seq.replace(".", "")

    # Convert to FASTA format
    fasta_chunks = []
    for seqname, seq in a3m_sequences.items():
        fasta_chunks.append(f">{seqname} {descriptions.get(seqname, '')}")
        fasta_chunks.append(seq)

    return "\n".join(fasta_chunks) + "\n"


class Nhmmer:
    """Python wrapper of the Nhmmer binary"""

    def __init__(
        self,
        binary_path,
        hmmalign_binary_path,
        hmmbuild_binary_path,
        database_path,
        n_cpu=8,
        e_value=1e-3,
        max_sequences=10000,
        alphabet="rna",
        time_limit_minutes=None,
    ):
        """Initialize Nhmmer wrapper"""
        self.binary_path = binary_path
        self.hmmalign_binary_path = hmmalign_binary_path
        self.hmmbuild_binary_path = hmmbuild_binary_path
        self.db_path = database_path
        self.e_value = e_value
        self.n_cpu = n_cpu
        self.max_sequences = max_sequences
        self.alphabet = alphabet
        self.time_limit_seconds = (
            time_limit_minutes * 60 if time_limit_minutes else None
        )

    def query(self, target_sequence):
        """Query the database using Nhmmer and return results in A3M format"""
        import os
        import subprocess
        import tempfile
        import time

        logging.info(f"Querying database with sequence: {target_sequence[:20]}...")

        with tempfile.TemporaryDirectory() as query_tmp_dir:
            input_fasta_path = os.path.join(query_tmp_dir, "query.fasta")
            output_sto_path = os.path.join(query_tmp_dir, "output.sto")

            # Create query FASTA file
            create_query_fasta_file(sequence=target_sequence, path=input_fasta_path)

            # Prepare Nhmmer command
            cmd_flags = [
                "-o",
                "/dev/null",  # Don't pollute stdout
                "--noali",  # Don't include the alignment in stdout
                "--cpu",
                str(self.n_cpu),
                "-E",
                str(self.e_value),
                "-A",
                output_sto_path,
            ]

            # Add alphabet flag
            if self.alphabet:
                cmd_flags.extend([f"--{self.alphabet}"])

            # Special handling for short RNA sequences
            if self.alphabet == "rna" and len(target_sequence) < SHORT_SEQUENCE_CUTOFF:
                cmd_flags.extend(["--F3", str(0.02)])
            else:
                cmd_flags.extend(["--F3", str(1e-5)])

            # Add input and database paths
            cmd_flags.extend([input_fasta_path, self.db_path])

            # Setup progress monitoring
            if self.time_limit_seconds is None:
                print(
                    f"⏳ Running Nhmmer search against {os.path.basename(self.db_path)} (no time limit)"
                )
            else:
                print(
                    f"⏳ Running Nhmmer search against {os.path.basename(self.db_path)} (time limit: {self.time_limit_seconds // 60} min)"
                )

            # Create a process with timeout
            cmd = [self.binary_path, *cmd_flags]
            start_time = time.time()

            try:
                # Use subprocess with timeout
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                # Create progress bar for time monitoring
                if self.time_limit_seconds is None:
                    # If no time limit, just wait for the process to finish
                    while process.poll() is None:
                        time.sleep(5)  # Check periodically
                        elapsed = time.time() - start_time
                        # Update progress every minute
                        if int(elapsed) % 60 == 0:
                            print(
                                f"⏳ Search in progress... (elapsed time: {elapsed:.0f} seconds)"
                            )
                else:
                    # If time limit is set, use a progress bar
                    with tqdm(
                        total=self.time_limit_seconds, desc="Search time", unit="sec"
                    ) as pbar:
                        elapsed = 0
                        while (
                            process.poll() is None and elapsed < self.time_limit_seconds
                        ):
                            time.sleep(1)
                            elapsed = time.time() - start_time
                            pbar.update(1)
                            pbar.set_description(
                                f"Search time ({elapsed:.0f}/{self.time_limit_seconds} sec)"
                            )

                        # If we hit the time limit, terminate the process
                        if process.poll() is None:
                            print(
                                f"⚠️ Time limit reached ({self.time_limit_seconds} seconds). Terminating search."
                            )
                            process.terminate()
                            process.wait()

                            # Even with timeout, check if we got partial results
                            if (
                                os.path.exists(output_sto_path)
                                and os.path.getsize(output_sto_path) > 0
                            ):
                                print("✅ Found partial results within the time limit.")
                            else:
                                print("❌ No results found within the time limit.")
                                return f">query\n{target_sequence}"

                # Get process status
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    process.wait()

                # Get stdout/stderr
                stdout, stderr = process.communicate()

                # Report completion
                elapsed = time.time() - start_time
                print(f"✅ Search completed in {elapsed:.2f} seconds")

                if process.returncode != 0 and not (
                    os.path.exists(output_sto_path)
                    and os.path.getsize(output_sto_path) > 0
                ):
                    print(f"❌ Nhmmer failed with error: {stderr}")
                    return f">query\n{target_sequence}"

            except Exception as e:
                print(f"❌ Error running Nhmmer: {e}")
                return f">query\n{target_sequence}"

            # Check if we got any hits
            if os.path.exists(output_sto_path) and os.path.getsize(output_sto_path) > 0:
                with open(output_sto_path) as f:
                    line_count = sum(1 for _ in f)
                print(f"✅ Found hits in Stockholm file ({line_count} lines)")

                # Build profile from query sequence
                print("🧬 Building HMM profile from query sequence...")
                hmmbuild = Hmmbuild(
                    binary_path=self.hmmbuild_binary_path, alphabet=self.alphabet
                )
                target_sequence_fasta = f">query\n{target_sequence}\n"
                profile = hmmbuild.build_profile_from_fasta(target_sequence_fasta)

                # Convert Stockholm to A3M
                print("📝 Converting Stockholm to A3M format...")
                a3m_out = convert_stockholm_to_a3m(
                    output_sto_path, max_sequences=self.max_sequences - 1
                )

                # Align hits to the query profile
                print("📊 Aligning sequences to query profile...")
                aligner = Hmmalign(binary_path=self.hmmalign_binary_path)
                aligned_a3m = aligner.align_sequences_to_profile(
                    profile=profile, sequences_a3m=a3m_out
                )

                # Get sequence count
                seq_count = aligned_a3m.count(">")
                print(f"🎯 Successfully aligned {seq_count} sequences")

                # Return A3M with query sequence first
                return "".join([target_sequence_fasta, aligned_a3m])
            print("⚠️ No hits found")
            # No hits - return only query sequence
            return f">query\n{target_sequence}"


class Hmmbuild:
    """Python wrapper for hmmbuild - construct HMM profiles from MSA"""

    def __init__(self, binary_path, alphabet=None):
        """Initialize Hmmbuild wrapper"""
        self.binary_path = binary_path
        self.alphabet = alphabet

    def build_profile_from_fasta(self, fasta):
        """Build an HMM profile from a FASTA string"""
        import os
        import re

        # Process FASTA to remove inserted residues (lowercase letters)
        sequences, descriptions = parse_fasta(fasta)
        lines = []
        for seq, desc in zip(sequences, descriptions):
            # Remove inserted residues (lowercase)
            seq = re.sub("[a-z]+", "", seq)
            lines.append(f">{desc}\n{seq}\n")
        msa = "".join(lines)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_msa_path = os.path.join(tmp_dir, "query.msa")
            output_hmm_path = os.path.join(tmp_dir, "output.hmm")

            with open(input_msa_path, "w") as f:
                f.write(msa)

            # Prepare command
            cmd_flags = ["--informat", "afa"]
            if self.alphabet:
                cmd_flags.append(f"--{self.alphabet}")

            cmd_flags.extend([output_hmm_path, input_msa_path])
            cmd = [self.binary_path, *cmd_flags]

            # Run hmmbuild
            run_command(cmd=cmd, cmd_name="Hmmbuild")

            # Read the output profile
            with open(output_hmm_path) as f:
                hmm = f.read()

            return hmm


class Hmmalign:
    """Python wrapper of the hmmalign binary"""

    def __init__(self, binary_path):
        """Initialize Hmmalign wrapper"""
        self.binary_path = binary_path

    def align_sequences_to_profile(self, profile, sequences_a3m):
        """Align sequences to a profile and return in A3M format"""
        import os

        # Process A3M to remove gaps
        sequences, descriptions = parse_fasta(sequences_a3m)
        lines = []
        for seq, desc in zip(sequences, descriptions):
            # Remove gaps
            seq = seq.replace("-", "")
            lines.append(f">{desc}\n{seq}\n")
        sequences_no_gaps_a3m = "".join(lines)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_profile = os.path.join(tmp_dir, "profile.hmm")
            input_seqs = os.path.join(tmp_dir, "sequences.a3m")
            output_a3m_path = os.path.join(tmp_dir, "output.a3m")

            with open(input_profile, "w") as f:
                f.write(profile)

            with open(input_seqs, "w") as f:
                f.write(sequences_no_gaps_a3m)

            # Prepare command
            cmd = [
                self.binary_path,
                "-o",
                output_a3m_path,
                "--outformat",
                "A2M",  # A2M is A3M in the HMMER suite
                input_profile,
                input_seqs,
            ]

            # Run hmmalign
            run_command(cmd=cmd, cmd_name="Hmmalign")

            # Read the aligned output
            with open(output_a3m_path, encoding="utf-8") as f:
                a3m = f.read()

            return a3m


class Msa:
    """Multiple Sequence Alignment container with methods for manipulating it"""

    def __init__(
        self, query_sequence, chain_poly_type, sequences, descriptions, deduplicate=True
    ):
        """Initialize MSA container"""
        import string

        if len(sequences) != len(descriptions):
            raise ValueError("The number of sequences and descriptions must match.")

        self.query_sequence = query_sequence
        self.chain_poly_type = chain_poly_type

        if not deduplicate:
            self.sequences = sequences
            self.descriptions = descriptions
        else:
            self.sequences = []
            self.descriptions = []
            # A replacement table that removes all lowercase characters
            deletion_table = str.maketrans("", "", string.ascii_lowercase)
            unique_sequences = set()
            for seq, desc in zip(sequences, descriptions):
                # Using string.translate is faster than re.sub('[a-z]+', '')
                sequence_no_deletions = seq.translate(deletion_table)
                if sequence_no_deletions not in unique_sequences:
                    unique_sequences.add(sequence_no_deletions)
                    self.sequences.append(seq)
                    self.descriptions.append(desc)

        # Make sure the MSA always has at least the query
        self.sequences = self.sequences or [query_sequence]
        self.descriptions = self.descriptions or ["Original query"]

        # Check if the 1st MSA sequence matches the query sequence
        if not self._sequences_are_feature_equivalent(
            self.sequences[0], query_sequence
        ):
            raise ValueError(
                f"First MSA sequence {self.sequences[0]} is not the query sequence {query_sequence}"
            )

    def _sequences_are_feature_equivalent(self, sequence1, sequence2):
        """Check if two sequences are equivalent (ignoring insertions)"""
        import re

        # For RNA, we can simply compare the uppercase versions
        if self.chain_poly_type == RNA_CHAIN:
            seq1_upper = re.sub("[a-z]+", "", sequence1)
            seq2_upper = re.sub("[a-z]+", "", sequence2)
            return seq1_upper == seq2_upper
        return sequence1 == sequence2  # Fallback for other types

    @classmethod
    def from_multiple_msas(cls, msas, deduplicate=True):
        """Initialize MSA from multiple MSAs"""
        if not msas:
            raise ValueError("At least one MSA must be provided.")

        query_sequence = msas[0].query_sequence
        chain_poly_type = msas[0].chain_poly_type
        sequences = []
        descriptions = []

        for msa in msas:
            if msa.query_sequence != query_sequence:
                raise ValueError(
                    f"Query sequences must match: {[m.query_sequence for m in msas]}"
                )
            if msa.chain_poly_type != chain_poly_type:
                raise ValueError(
                    f"Chain poly types must match: {[m.chain_poly_type for m in msas]}"
                )
            sequences.extend(msa.sequences)
            descriptions.extend(msa.descriptions)

        return cls(
            query_sequence=query_sequence,
            chain_poly_type=chain_poly_type,
            sequences=sequences,
            descriptions=descriptions,
            deduplicate=deduplicate,
        )

    @classmethod
    def from_a3m(
        cls, query_sequence, chain_poly_type, a3m, max_depth=None, deduplicate=True
    ):
        """Parse a single A3M and build the Msa object"""
        sequences, descriptions = parse_fasta(a3m)

        if max_depth is not None and 0 < max_depth < len(sequences):
            print(
                f"MSA cropped from depth of {len(sequences)} to {max_depth} for {query_sequence}"
            )
            sequences = sequences[:max_depth]
            descriptions = descriptions[:max_depth]

        return cls(
            query_sequence=query_sequence,
            chain_poly_type=chain_poly_type,
            sequences=sequences,
            descriptions=descriptions,
            deduplicate=deduplicate,
        )

    @property
    def depth(self):
        """Return the number of sequences in the MSA"""
        return len(self.sequences)

    def to_a3m(self):
        """Return the MSA in A3M format"""
        a3m_lines = []
        for desc, seq in zip(self.descriptions, self.sequences):
            a3m_lines.append(f">{desc}")
            a3m_lines.append(seq)
        return "\n".join(a3m_lines) + "\n"


def af3_generate_rna_msa(rna_sequence, database_settings, afdb_dir, hmmer_path):
    """Generate MSA for an RNA sequence using AlphaFold3's pipeline"""
    import os
    import time

    # Validate RNA sequence
    rna_sequence = rna_sequence.upper().strip()
    valid_bases = set("ACGU")
    if not all(base in valid_bases for base in rna_sequence):
        raise ValueError(
            f"Invalid RNA sequence. Must contain only A, C, G, U: {rna_sequence}"
        )

    print(
        f"🧪 Validating RNA sequence: {rna_sequence[:20]}..."
        + ("" if len(rna_sequence) <= 20 else f"... ({len(rna_sequence)} nucleotides)")
    )

    # Setup paths to binaries and databases
    nhmmer_binary = os.path.join(hmmer_path, "bin/nhmmer")
    hmmalign_binary = os.path.join(hmmer_path, "bin/hmmalign")
    hmmbuild_binary = os.path.join(hmmer_path, "bin/hmmbuild")

    database_paths = {
        "Rfam": os.path.join(
            afdb_dir, "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
        ),
        "RNAcentral": os.path.join(
            afdb_dir, "rnacentral_active_seq_id_90_cov_80_linclust.fasta"
        ),
        "NT_RNA": os.path.join(
            afdb_dir, "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
        ),
    }

    # Download selected databases if they don't exist
    selected_dbs = []
    if database_settings.get("Rfam", False):
        selected_dbs.append("Rfam")
    if database_settings.get("RNAcentral", False):
        selected_dbs.append("RNAcentral")
    if database_settings.get("NT_RNA", False):
        selected_dbs.append("NT_RNA")

    # Download any missing databases that are selected
    missing_dbs = []
    for db_key in selected_dbs:
        db_filename = None
        for filename, key in RNA_DATABASE_INFO.items():
            if key == db_key:
                db_filename = filename
                break

        if db_filename:
            db_path = os.path.join(afdb_dir, db_filename)
            if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
                missing_dbs.append(db_key)

    if missing_dbs:
        print(f"⚠️ Some selected databases are missing: {', '.join(missing_dbs)}")
        print("📥 Downloading missing databases...")
        download_selected_databases(database_settings, afdb_dir)
        print("Continuing with pipeline execution...")

    # Check which databases actually exist now
    existing_dbs = []
    with tqdm(total=len(database_paths), desc="Checking databases", unit="db") as pbar:
        for db_name, db_path in database_paths.items():
            if os.path.exists(db_path) and os.path.getsize(db_path) > 0:
                db_key = RNA_DATABASE_INFO.get(os.path.basename(db_path))
                if db_key:
                    existing_dbs.append(db_key)
            pbar.update(1)

    # Check if any selected database exists
    selected_existing_dbs = [
        db for db in existing_dbs if database_settings.get(db, False)
    ]

    if not selected_existing_dbs:
        print(
            "⚠️ No selected databases could be downloaded or found. Continuing with query sequence only."
        )
        return f">query\n{rna_sequence}\n"
    print(
        f"📊 Found {len(selected_existing_dbs)} selected databases: {', '.join(selected_existing_dbs)}"
    )

    # Generate MSA
    print("🚀 Starting MSA generation...")

    print(f"Getting RNA MSAs for sequence: {rna_sequence[:20]}...")
    print(f"🧬 Starting MSA search for RNA sequence of length {len(rna_sequence)}...")
    rna_msa_start_time = time.time()

    # Extract settings
    time_limit_minutes = database_settings.get("time_limit_minutes")
    max_sequences_per_db = database_settings.get("max_sequences_per_db", 10000)
    n_cpu = database_settings.get("n_cpu", 2)  # Default to 2 for Colab
    e_value = database_settings.get("e_value", 0.001)

    # Filter database paths based on settings and check if files exist
    filtered_db_paths = {}
    for db_name, db_path in database_paths.items():
        db_key = None
        for file_name, key in RNA_DATABASE_INFO.items():
            if file_name in db_path:
                db_key = key
                break

        # Check if database is selected in settings AND file exists with content
        if (
            db_key
            and database_settings.get(db_key, False)
            and os.path.exists(db_path)
            and os.path.getsize(db_path) > 0
        ):
            filtered_db_paths[db_name] = db_path

    # Setup progress tracking
    total_dbs = len(filtered_db_paths)
    if total_dbs == 0:
        print("❌ No selected databases found or none selected in settings.")
        return f">query\n{rna_sequence}\n"

    time_limit_str = (
        "no time limit"
        if time_limit_minutes is None
        else f"{time_limit_minutes} minutes per database"
    )
    print(f"🔍 Will search {total_dbs} databases with {time_limit_str}")
    progress_bar = tqdm(total=total_dbs, desc="Database searches", unit="db")

    # Run Nhmmer on each database
    msas = []
    for db_name, db_path in filtered_db_paths.items():
        print(f"🔍 Searching database: {os.path.basename(db_path)}...")
        nhmmer_runner = Nhmmer(
            binary_path=nhmmer_binary,
            hmmalign_binary_path=hmmalign_binary,
            hmmbuild_binary_path=hmmbuild_binary,
            database_path=db_path,
            n_cpu=n_cpu,
            e_value=e_value,
            max_sequences=max_sequences_per_db,
            alphabet="rna",
            time_limit_minutes=time_limit_minutes,
        )

        try:
            a3m_result = nhmmer_runner.query(rna_sequence)
            msa = Msa.from_a3m(
                query_sequence=rna_sequence,
                chain_poly_type=RNA_CHAIN,
                a3m=a3m_result,
                deduplicate=False,
            )
            msas.append(msa)
            print(f"✅ Found {msa.depth} sequences in {db_name}")
            progress_bar.update(1)
        except Exception as e:
            print(f"❌ Error processing {db_name}: {e}")
            progress_bar.update(1)

    progress_bar.close()

    # Merge and deduplicate MSAs
    print("🔄 Merging and deduplicating sequences from all databases...")
    if not msas:
        # If all searches failed, create an empty MSA with just the query
        rna_msa = Msa(
            query_sequence=rna_sequence,
            chain_poly_type=RNA_CHAIN,
            sequences=[rna_sequence],
            descriptions=["Original query"],
            deduplicate=False,
        )
        print("⚠️ No homologous sequences found. MSA contains only the query sequence.")
        a3m = f">query\n{rna_sequence}\n"
    else:
        rna_msa = Msa.from_multiple_msas(msas=msas, deduplicate=True)
        print(f"🎉 MSA construction complete! Found {rna_msa.depth} unique sequences.")
        a3m = rna_msa.to_a3m()

    elapsed_time = time.time() - rna_msa_start_time
    print(f"⏱️ Total MSA generation time: {elapsed_time:.2f} seconds")

    return a3m


def process_yaml_files(
    yaml_dir,
    af_input_dir,
    af_input_apo_dir,
    binder_chain="A",
    use_msa_for_af3=False,
):
    """
    Process YAML files for AlphaFold3 input, supporting flexible multi-chain, multi-type targets.

    Args:
        yaml_dir (Path): Directory containing YAML files
        af_input_dir (str): Output directory for complex JSON files
        af_input_apo_dir (str): Output directory for apo JSON files
        target_type (str): Not actually used, will be ignored for flexibility
        binder_chain (str): Binder chain (default "A")
        mod_to_wt_aa, afdb_dir, hmmer_path, use_msa_for_af3: see prior API
    """
    # For each design YAML, parse the "sequences" flexibly: detect proteins, ligands, nucleics, metals, etc!
    for yaml_path in Path(yaml_dir).glob("*.yaml"):
        print("yaml_path", yaml_path)
        name = os.path.basename(yaml_path).split(".yaml")[0]
        with open(yaml_path) as file:
            yaml_data = yaml.safe_load(file)

        # Parse out all chains and types in the data for this YAML
        protein_entries = []
        ligand_entries = []
        rna_entries = []
        dna_entries = []
        metal_entries = []
        binder_protein_entry = None
        binder_idx = None

        for i, seq in enumerate(yaml_data["sequences"]):
            # Proteins
            if "protein" in seq:
                ids = seq["protein"].get("id", ["?"])
                if binder_chain in ids:
                    binder_idx = i
                    binder_protein_entry = seq["protein"]
                else:
                    protein_entries.append((i, seq["protein"]))
            # Small-molecule ligand or metal (ccd can mean either)
            if "ligand" in seq:
                entry = seq["ligand"]
                if entry.get("smiles"):
                    ligand_entries.append(entry)
                elif entry.get("ccd"):
                    # Could be metal or small-molecule, we'll just use ccd "as is"
                    metal_entries.append(entry)
            # RNA or DNA
            if "rna" in seq:
                rna_entries.append(seq["rna"])
            if "dna" in seq:
                dna_entries.append(seq["dna"])

        # Build JSON entries for all present types
        # Proteins: order is all "target" proteins (non-binder), then binder
        all_protein_ids = []
        all_protein_seqs = []
        all_protein_msas = []
        modification_ls = []
        modification_chain = []
        query_seqs = []
        binder_chain_id = None
        # First, all non-binder proteins
        for _, p in protein_entries:
            all_protein_ids.append(p["id"][0])
            all_protein_seqs.append(p["sequence"])
            all_protein_msas.append(p.get("msa", "empty"))
            query_seqs.append(p["sequence"])
            if "modifications" in p:
                modification_ls.append(p["modifications"])
                modification_chain.append(p["id"][0])
            else:
                modification_ls.append([])
                modification_chain.append([])
        # The binder protein (last)
        if binder_protein_entry:
            binder_chain_id = binder_protein_entry["id"][0]
            all_protein_ids.append(binder_chain_id)
            all_protein_seqs.append(binder_protein_entry["sequence"])
            all_protein_msas.append(binder_protein_entry.get("msa", "empty"))
            query_seqs.append(binder_protein_entry["sequence"])
            if "modifications" in binder_protein_entry:
                modification_ls.append(binder_protein_entry["modifications"])
                modification_chain.append(binder_chain_id)
            else:
                modification_ls.append([])
                modification_chain.append([])

        # Process MSAs for each protein; ensure each gets a formatted sequence
        processed_msas = []
        for target_seq, msa_path, chain_id, query_seq in zip(
            all_protein_seqs, all_protein_msas, all_protein_ids, query_seqs
        ):
            if chain_id == binder_chain:
                # For the binder chain, do not get MSA, just insert the query sequence
                processed_msas.append(f">query\n{query_seq}")
                continue

            msa_file = None
            if msa_path and msa_path not in ["empty", ""]:
                msa_file = msa_path.replace(".npz", ".a3m")
                if os.path.exists(msa_file):
                    protein_msa = extract_sequences_and_format(
                        msa_file, replace_query_seq=True, query_seq=query_seq
                    )
                else:
                    protein_msa = f">query\n{query_seq}"
                processed_msas.append(protein_msa)
            elif msa_path == "empty" and use_msa_for_af3:
                try:
                    # Best effort: process MSA on-the-fly
                    msa_file = process_msa(
                        chain_id=chain_id,
                        sequence=target_seq,
                        pdb_code="query",
                        msa_dir=Path(yaml_dir),
                    )
                    protein_msa = extract_sequences_and_format(
                        msa_file, replace_query_seq=True, query_seq=query_seq
                    )
                    processed_msas.append(protein_msa)
                except Exception as e:
                    print(f"WARNING: Failed to get or make MSA for {chain_id}: {e}")
                    processed_msas.append(f">query\n{query_seq}")
            else:
                processed_msas.append(f">query\n{query_seq}")

        # Build the json_result as needed for AF3
        json_result = {
            "name": name,
            "sequences": [],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
        }
        # Add proteins
        for i, (seq, chain) in enumerate(zip(all_protein_seqs, all_protein_ids)):
            protein_entry = {
                "protein": {
                    "id": chain,
                    "sequence": seq,
                    "pairedMsa": processed_msas[i],
                    "unpairedMsa": processed_msas[i],
                    "templates": [],
                }
            }
            # Add modifications if present
            if modification_chain and i < len(modification_chain):
                mods = modification_ls[i]
                if len(mods) > 0:
                    protein_entry["protein"]["modifications"] = []
                    for item in mods:
                        protein_entry["protein"]["modifications"].append(
                            {
                                "ptmType": item.get("ccd") or item.get("ptmType"),
                                "ptmPosition": item.get("position")
                                or item.get("ptmPosition"),
                            }
                        )
            json_result["sequences"].append(protein_entry)
        # Add rna
        for rna in rna_entries:
            rna_entry = {
                "rna": {
                    "id": rna["id"][0] if isinstance(rna["id"], list) else rna["id"],
                    "sequence": rna["sequence"],
                    "unpairedMsa": f">query\n{rna['sequence']}",
                }
            }
            json_result["sequences"].append(rna_entry)
        # Add dna
        for dna in dna_entries:
            dna_entry = {
                "dna": {
                    "id": dna["id"][0] if isinstance(dna["id"], list) else dna["id"],
                    "sequence": dna["sequence"],
                }
            }
            json_result["sequences"].append(dna_entry)
        # Add ligands (SMILES and/or ccds; ccd can be used for metal too)
        for lig in ligand_entries:
            lig_entry = {
                "ligand": {
                    "id": lig["id"][0] if isinstance(lig["id"], list) else lig["id"],
                    "smiles": lig["smiles"],
                }
            }
            json_result["sequences"].append(lig_entry)
        for lig in metal_entries:
            lig_entry = {
                "ligand": {
                    "id": lig["id"][0] if isinstance(lig["id"], list) else lig["id"],
                    "ccdCodes": [lig["ccd"]],
                }
            }
            json_result["sequences"].append(lig_entry)

        # Build the json_result_apo: only the binder (or last protein entry)
        # fallback: if no proteins, skip apo!
        json_result_apo = {
            "name": name,
            "sequences": [],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1,
        }
        if len(all_protein_ids) > 0 and len(all_protein_seqs) > 0:
            i_binder = len(all_protein_ids) - 1
            seq = all_protein_seqs[i_binder]
            chain = all_protein_ids[i_binder]
            protein_entry = {
                "protein": {
                    "id": chain,
                    "sequence": seq,
                    "pairedMsa": processed_msas[i_binder]
                    if processed_msas
                    else f">query\n{seq}",
                    "unpairedMsa": processed_msas[i_binder]
                    if processed_msas
                    else f">query\n{seq}",
                    "templates": [],
                }
            }
            # Add modifications if present
            if modification_chain and i_binder < len(modification_chain):
                mods = modification_ls[i_binder]
                if len(mods) > 0:
                    protein_entry["protein"]["modifications"] = []
                    for item in mods:
                        protein_entry["protein"]["modifications"].append(
                            {
                                "ptmType": item.get("ccd") or item.get("ptmType"),
                                "ptmPosition": item.get("position")
                                or item.get("ptmPosition"),
                            }
                        )
            json_result_apo["sequences"].append(protein_entry)

        # Write files
        with open(os.path.join(af_input_dir, f"{name}.json"), "w") as f:
            json.dump(json_result, f, indent=4)

        with open(os.path.join(af_input_apo_dir, f"{name}.json"), "w") as f:
            json.dump(json_result_apo, f, indent=4)


def extract_sequences_and_format(a3m_file_path, replace_query_seq=False, query_seq=""):
    sequences = []
    first_sequence_reformatted = (
        replace_query_seq  # Track if we have renamed the first sequence
    )
    print("a3m_file_path", a3m_file_path)
    with open(a3m_file_path) as file:
        seq_id = None
        seq_data = ""
        for line in file:
            line = line.strip()
            # Skip comment lines or empty lines
            if line.startswith("#") or not line:
                continue

            if line.startswith(">"):
                if seq_id:
                    if replace_query_seq and not first_sequence_reformatted:
                        seq_id = "wild-type"
                        first_sequence_reformatted = True
                    sequences.append(f">{seq_id}\n{seq_data}")

                seq_id = line[1:]
                seq_data = ""
            else:
                seq_data += line
        if seq_id:
            if replace_query_seq and not first_sequence_reformatted:
                seq_id = "wild-type"
            sequences.append(f">{seq_id}\n{seq_data}")

    if replace_query_seq:
        query_seq_formatted = f">query\n{query_seq}"
        sequences.insert(0, query_seq_formatted)

    joined_string = "\n".join(sequences)
    return joined_string
