import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ['X']

    def res_1to3(r: int) -> str:
        return residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types
    pdb_lines: List[str] = []
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index
    if np.any(aatype > residue_constants.restype_num):
        raise ValueError('Invalid aatypes.')
    headers = get_pdb_headers(prot)
    if len(headers) > 0:
        pdb_lines.extend(headers)
    n = aatype.shape[0]
    atom_index = 1
    prev_chain_index = 0
    chain_tags = string.ascii_uppercase
    chain_tag = None
    for i in range(n):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue
            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.0
            element = atom_name[0]
            charge = ''
            chain_tag = 'A'
            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]
            atom_line = f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}{res_name_3:>3} {chain_tag:>1}{residue_index[i]:>4}{insertion_code:>1}   {pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}          {element:>2}{charge:>2}'
            pdb_lines.append(atom_line)
            atom_index += 1
        should_terminate = i == n - 1
        if chain_index is not None:
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]
        if should_terminate:
            chain_end = 'TER'
            chain_termination_line = f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[i]):>3} {chain_tag:>1}{residue_index[i]:>4}'
            pdb_lines.append(chain_termination_line)
            atom_index += 1
            if i != n - 1:
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))
    pdb_lines.append('END')
    pdb_lines.append('')
    return '\n'.join(pdb_lines)