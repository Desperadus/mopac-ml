#!/usr/bin/env python3
"""
Convert PDB/CIF to a MOPAC-ML (.in) file with:
- Cleaning (chains/ligands/waters) via Bio.PDB
- Protonation & per-atom charges via PDB2PQR at a chosen pH
- Automatic total charge detection from the PQR
- Robust handling of multi-molecule XYZ output from Open Babel (flattened)

Requirements:
  - biopython
  - openbabel (CLI: obabel)
  - pdb2pqr (CLI: pdb2pqr)  # e.g., conda install -c conda-forge pdb2pqr

Usage example:
  python make_mopac_in.py input.pdb --ph 7.4 -o mopac.in
  # or override auto-charge:
  python make_mopac_in.py input.pdb --ph 7.4 --charge -3
"""

import argparse
import subprocess
import sys
import tempfile
import os
import re
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select


class StructureCleaner(Select):
    """
    Filters a structure:
    - Optionally restrict to a single chain.
    - Optionally keep only listed ligand residue names.
    - Optionally remove water.
    """

    def __init__(self, chain_id, keep_ligands=None, remove_water=False):
        self.chain_id = chain_id
        self.keep_ligands = set(keep_ligands) if keep_ligands else None
        self.remove_water = remove_water

    def accept_chain(self, chain):
        return True if self.chain_id is None else (chain.id == self.chain_id)

    def accept_residue(self, residue):
        resname = residue.get_resname()
        # amino acids, nucleotides, etc.
        is_standard_residue = residue.id[0] == ' '
        is_water = (resname == 'HOH')

        if self.remove_water and is_water:
            return False

        if is_standard_residue:
            return True

        # HETATM: keep if no ligand filter provided (keep all),
        # or if explicitly requested in --ligands list.
        if self.keep_ligands is None:
            return True
        return resname in self.keep_ligands


def check_dependencies():
    """Checks for obabel, pdb2pqr, and BioPython."""
    try:
        subprocess.run(["obabel", "-V"], check=True,
                       capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'obabel' (Open Babel) not found in PATH.", file=sys.stderr)
        print("Install e.g.: conda install -c conda-forge openbabel", file=sys.stderr)
        sys.exit(1)

    try:
        subprocess.run(["pdb2pqr", "--version"], check=True,
                       capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'pdb2pqr' not found in PATH.", file=sys.stderr)
        print("Install e.g.: conda install -c conda-forge pdb2pqr", file=sys.stderr)
        sys.exit(1)

    try:
        import Bio.PDB  # noqa: F401
    except ImportError:
        print("Error: 'biopython' not found.", file=sys.stderr)
        print("Install e.g.: pip install biopython", file=sys.stderr)
        sys.exit(1)


def _flatten_xyz_blocks(xyz_text):
    """
    Open Babel may emit multiple XYZ 'molecules':
      <N>\\n<comment>\\n<atoms x N>\\n <N>\\n<comment>\\n<atoms x N>...
    This concatenates ALL atom lines into one continuous block,
    discarding every per-molecule header/comment.
    """
    lines = [ln.strip()
             for ln in xyz_text.strip().splitlines() if ln.strip() != ""]
    i = 0
    out_atom_lines = []
    L = len(lines)

    while i < L:
        # Expect integer atom count
        try:
            n = int(lines[i])
        except ValueError:
            # If the first line isn't a number, assume we already have bare atoms
            # (some tools can output naked XYZ blocks); fall back to returning all lines.
            return "\n".join(lines)
        i += 1  # move to comment line
        if i >= L:
            break
        # skip comment line
        i += 1

        # collect n atom lines
        block_atoms = lines[i:i + n]
        if len(block_atoms) != n:
            raise ValueError(
                "Malformed XYZ: atom count does not match lines present.")
        out_atom_lines.extend(block_atoms)
        i += n

    if not out_atom_lines:
        raise ValueError("Open Babel produced empty XYZ output.")
    return "\n".join(out_atom_lines)


def run_pdb2pqr(input_pdb, ph=None, ff="amber"):
    """
    Run PDB2PQR on a PDB file. Returns path to the generated PQR file.
    """
    out_fd, out_path = tempfile.mkstemp(suffix=".pqr")
    os.close(out_fd)  # we'll let pdb2pqr write it

    cmd = ["pdb2pqr", f"--ff={ff}", input_pdb, out_path]
    # Add pH handling if specified
    if ph is not None:
        cmd.insert(1, f"--with-ph={ph}")

    print(f"Running PDB2PQR ({' '.join(cmd)}) ...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out_path
    except subprocess.CalledProcessError as e:
        print("Error running 'pdb2pqr':", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        # Clean up
        try:
            os.remove(out_path)
        except OSError:
            pass
        sys.exit(1)


_float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_total_charge_from_pqr(pqr_path):
    """
    Sum the per-atom charges in a PQR (ATOM/HETATM lines). The charge is the
    penultimate float on each line: x y z  charge  radius
    Returns (charge_sum, charge_int)
    """
    total = 0.0
    n_atoms = 0
    with open(pqr_path, "r") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            # Extract all floats, take the last two as (charge, radius)
            floats = _float_re.findall(line)
            if len(floats) < 5:
                # Unexpected line; skip conservatively
                continue
            charge = float(floats[-2])
            total += charge
            n_atoms += 1
    if n_atoms == 0:
        raise ValueError(
            "No ATOM/HETATM records found in PQR — cannot determine charge.")
    charge_int = int(round(total))
    return total, charge_int, n_atoms


def xyz_from_pqr_with_obabel(pqr_path):
    """
    Use Open Babel to convert PQR -> XYZ, then flatten multi-block XYZ if needed.
    """
    cmd = ["obabel", "-ipqr", pqr_path, "-oxyz"]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        return _flatten_xyz_blocks(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running Open Babel ('obabel'):", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


def create_mopac_input(xyz_atom_lines, charge, solvation=False,
                       calc_type='GRADIENTS', threads=1):
    """
    Build a valid MOPAC input:
      line 1: keywords
      line 2: title
      line 3: comment
      lines 4+: geometry in XYZ format (element x y z)
    """
    keywords = ["MOZYME", "PM6", "MLCORR", "XYZ", "SINGLET", "RHF"]

    if calc_type == '1SCF':
        keywords.append("1SCF")
    elif calc_type == 'Optimize':
        keywords.append("GEO-OK")
    elif calc_type == 'GRADIENTS':
        keywords.append("GRADIENTS")

    keywords.append(f"THREADS={threads}")
    if solvation:
        keywords.append("COSMO")

    # Always-on diagnostics
    keywords.extend(["NOMM", "AUX(MOS=-99999,XW,XS,XP,PRECISION=6)"])

    title = "Generated by make_mopac_in.py"
    comment = f"Auto CHARGE={charge}; atoms={len(xyz_atom_lines.splitlines())}"

    return f"{' '.join(keywords)}\n\n\n{xyz_atom_lines}\n"


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDB/CIF to a MOPAC-ML (.in) file using PDB2PQR for protonation and automatic total charge.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- IO ---
    parser.add_argument("input_file", help="Input PDB or CIF file.")
    parser.add_argument("-o", "--output", default="mopac.in",
                        help="Output MOPAC input file name (default: mopac.in)")

    # --- Filtering ---
    parser.add_argument("-c", "--chain", default=None,
                        help="Chain ID to keep (e.g., 'A'). Default: keep all chains.")
    parser.add_argument("-l", "--ligands", nargs='+', default=None,
                        help="Residue names of ligands to KEEP (e.g., LIG UNL). Default: keep all HETATMs.")
    parser.add_argument("--remove_water", action="store_true",
                        help="If set, remove waters (HOH). Default: keep waters.")

    # --- Calculation params ---
    parser.add_argument("--charge", type=int, default=None,
                        help="Total charge override (e.g., -2). If omitted, sum of PQR charges is used.")
    parser.add_argument("--ph", type=float, default=7.4,
                        help="Target pH for PDB2PQR protonation (default: 7.4).")
    parser.add_argument("--ff", type=str, default="AMBER",
                        help="PDB2PQR force field (default: amber).")
    parser.add_argument("-s", "--solvation", action='store_true',
                        help="Add implicit solvation (COSMO).")
    parser.add_argument("--calc_type", choices=['1SCF', 'Optimize', 'GRADIENTS'],
                        default='1SCF',
                        help="Type: '1SCF' (single-point), 'Optimize' (geometry opt), 'GRADIENTS' (forces). Default: GRADIENTS")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of CPU threads (default: 1).")

    args = parser.parse_args()

    # 0) Dependencies
    print("Checking dependencies...")
    check_dependencies()

    # 1) Load structure
    print(f"Loading structure from {args.input_file}...")
    ext = os.path.splitext(args.input_file)[1].lower()
    parser_cls = MMCIFParser if ext == ".cif" else PDBParser

    try:
        structure = parser_cls(QUIET=True).get_structure(
            "struc", args.input_file)
    except Exception as e:
        print(f"Error parsing structure file: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Filter structure and write a temporary cleaned PDB
    print(
        f"Filtering structure (Chain: {args.chain}, Keep ligands: {args.ligands}, Remove water: {args.remove_water})...")
    cleaner = StructureCleaner(
        args.chain, args.ligands, remove_water=args.remove_water)
    io = PDBIO()
    io.set_structure(structure)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        temp_pdb = tmp.name
    try:
        io.save(temp_pdb, cleaner)

        # 3) PDB2PQR: protonate (at pH) and assign charges -> PQR
        pqr_path = run_pdb2pqr(temp_pdb, ph=args.ph, ff=args.ff)

        # 4) Determine total charge from PQR if not overridden
        q_sum, q_int, n_atoms = parse_total_charge_from_pqr(pqr_path)
        if args.charge is None:
            total_charge = q_int
            print(
                f"Auto-detected total charge from PQR: sum={q_sum:.4f} over {n_atoms} atoms; using CHARGE={q_int}")
        else:
            total_charge = args.charge
            print(
                f"Using user-specified CHARGE={total_charge} (PQR sum was {q_sum:.4f} over {n_atoms} atoms)")

        # 5) Convert PQR -> XYZ (preserves added H) and flatten
        xyz_atom_lines = xyz_from_pqr_with_obabel(pqr_path)

    finally:
        # cleanup temporary PDB and PQR
        try:
            os.remove(temp_pdb)
        except OSError:
            pass
        try:
            if 'pqr_path' in locals() and os.path.exists(pqr_path):
                os.remove(pqr_path)
        except OSError:
            pass

    # 6) Build MOPAC input
    print("Assembling MOPAC input file...")
    mopac_text = create_mopac_input(
        xyz_atom_lines=xyz_atom_lines,
        charge=total_charge,
        solvation=args.solvation,
        calc_type=args.calc_type,
        threads=args.threads
    )

    # 7) Write output
    with open(args.output, "w") as f:
        f.write(mopac_text)

    print(f"\nSuccess! MOPAC input file written to: {args.output}")
    print(f"  Keywords: {mopac_text.splitlines()[0]}")
    print("\nNext step: Run 'mopac_ml {output}'".format(output=args.output))


if __name__ == "__main__":
    main()
