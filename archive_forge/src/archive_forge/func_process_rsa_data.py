import os
import tempfile
import shutil
import subprocess
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.AbstractPropertyMap import (
def process_rsa_data(rsa_data):
    """Process the .rsa output file: residue level SASA data."""
    naccess_rel_dict = {}
    for line in rsa_data:
        if line.startswith('RES'):
            res_name = line[4:7]
            chain_id = line[8]
            resseq = int(line[9:13])
            icode = line[13]
            res_id = (' ', resseq, icode)
            naccess_rel_dict[chain_id, res_id] = {'res_name': res_name, 'all_atoms_abs': float(line[16:22]), 'all_atoms_rel': float(line[23:28]), 'side_chain_abs': float(line[29:35]), 'side_chain_rel': float(line[36:41]), 'main_chain_abs': float(line[42:48]), 'main_chain_rel': float(line[49:54]), 'non_polar_abs': float(line[55:61]), 'non_polar_rel': float(line[62:67]), 'all_polar_abs': float(line[68:74]), 'all_polar_rel': float(line[75:80])}
    return naccess_rel_dict