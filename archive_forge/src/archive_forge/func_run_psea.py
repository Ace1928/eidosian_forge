import subprocess
import os
from Bio.PDB.Polypeptide import is_aa
def run_psea(fname, verbose=False):
    """Run PSEA and return output filename.

    Note that this assumes the P-SEA binary is called "psea" and that it is
    on the path.

    Note that P-SEA will write an output file in the current directory using
    the input filename with extension ".sea".

    Note that P-SEA will not write output to the terminal while run unless
     verbose is set to True.
    """
    last = fname.split('/')[-1]
    base = last.split('.')[0]
    cmd = ['psea', fname]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(p.stdout)
    if not p.stderr.strip() and os.path.exists(base + '.sea'):
        return base + '.sea'
    else:
        raise RuntimeError(f'Error running p-sea: {p.stderr}')