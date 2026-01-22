import csv
import itertools
import logging
import math
import re
import sys
from collections import defaultdict, namedtuple
from typing import Generator, List
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, molzip
from rdkit.Chem import rdRGroupDecomposition as rgd
def molzip_smi(smiles):
    """Fix a rgroup smiles for molzip, note that the core MUST come first
    in the smiles string, ala core.rgroup1.rgroup2 ...
    """
    dupes = set()
    sl = []
    for s in smiles.split('.'):
        if s.count('*') >= 1:
            if s in dupes:
                continue
            else:
                dupes.add(s)
        sl.append(s)
    smiles = '.'.join(sl)
    m = Chem.RWMol(Chem.MolFromSmiles(smiles, sanitize=False))
    frags = Chem.GetMolFrags(m)
    core = frags[0]
    atommaps = {}
    counts = defaultdict(int)
    for idx in core:
        atommap = m.GetAtomWithIdx(idx).GetAtomMapNum()
        if atommap:
            atommaps[atommap] = idx
            counts[atommap] += 1
    next_atommap = max(atommaps) + 1
    add_atommap = []
    for fragment in frags[1:]:
        for idx in fragment:
            atommap = m.GetAtomWithIdx(idx).GetAtomMapNum()
            if atommap:
                count = counts[atommap] = counts[atommap] + 1
                if count > 2:
                    m.GetAtomWithIdx(idx).SetAtomMapNum(next_atommap)
                    add_atommap.append((atommaps[atommap], next_atommap))
                    next_atommap += 1
    for atomidx, atommap in add_atommap:
        atom = m.GetAtomWithIdx(atomidx)
        bonds = list(atom.GetBonds())
        if len(bonds) == 1:
            oatom = bonds[0].GetOtherAtom(atom)
            xatom = Chem.Atom(0)
            idx = m.AddAtom(xatom)
            xatom = m.GetAtomWithIdx(idx)
            xatom.SetAtomMapNum(atommap)
            m.AddBond(oatom.GetIdx(), xatom.GetIdx(), Chem.BondType.SINGLE)
    return Chem.molzip(m)