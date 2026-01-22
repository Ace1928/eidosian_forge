import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
def from_proteinnet_string(proteinnet_str: str) -> Protein:
    tag_re = '(\\[[A-Z]+\\]\\n)'
    tags: List[str] = [tag.strip() for tag in re.split(tag_re, proteinnet_str) if len(tag) > 0]
    groups: Iterator[Tuple[str, List[str]]] = zip(tags[0::2], [l.split('\n') for l in tags[1::2]])
    atoms: List[str] = ['N', 'CA', 'C']
    aatype = None
    atom_positions = None
    atom_mask = None
    for g in groups:
        if '[PRIMARY]' == g[0]:
            seq = g[1][0].strip()
            for i in range(len(seq)):
                if seq[i] not in residue_constants.restypes:
                    seq[i] = 'X'
            aatype = np.array([residue_constants.restype_order.get(res_symbol, residue_constants.restype_num) for res_symbol in seq])
        elif '[TERTIARY]' == g[0]:
            tertiary: List[List[float]] = []
            for axis in range(3):
                tertiary.append(list(map(float, g[1][axis].split())))
            tertiary_np = np.array(tertiary)
            atom_positions = np.zeros((len(tertiary[0]) // 3, residue_constants.atom_type_num, 3)).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_positions[:, residue_constants.atom_order[atom], :] = np.transpose(tertiary_np[:, i::3])
            atom_positions *= PICO_TO_ANGSTROM
        elif '[MASK]' == g[0]:
            mask = np.array(list(map({'-': 0, '+': 1}.get, g[1][0].strip())))
            atom_mask = np.zeros((len(mask), residue_constants.atom_type_num)).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_mask[:, residue_constants.atom_order[atom]] = 1
            atom_mask *= mask[..., None]
    assert aatype is not None
    return Protein(atom_positions=atom_positions, atom_mask=atom_mask, aatype=aatype, residue_index=np.arange(len(aatype)), b_factors=None)