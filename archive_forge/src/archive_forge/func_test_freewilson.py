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
def test_freewilson():
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    assert dummypat.findall('C[*:1]N.[H][*:2]') == ['1', '2']
    assert dummypat.findall('C[*:1]N.[HH][*:2]') == ['1', '2']
    assert dummypat.findall('C[*:1]N.[2H][*:2]') == ['1', '2']
    assert dummypat.findall('C[*:1]N.[CH2][*:2]') == ['1', '2']
    scaffold = Chem.MolFromSmiles('[*:2]c1cccnc1[*:1]')
    mols = [Chem.MolFromSmiles('N' * (i + 1) + 'c1cccnc1' + 'C' * (i + 1)) for i in range(10)]
    scores = [Descriptors.MolLogP(m) for m in mols]
    fw = FWDecompose(scaffold, mols, scores)
    for pred in FWBuild(fw, pred_filter=lambda x: -3 < x < 3, mw_filter=lambda mw: 100 < mw < 450, hvy_filter=lambda hvy: 10 < hvy < 50, mol_filter=lambda m: -3 < Descriptors.MolLogP(m) < 3):
        rgroups = set()
        for sidechain in pred.rgroups:
            rgroups.add(sidechain.rgroup)
        rgroups = sorted(rgroups)
        assert list(rgroups) == ['Core', 'R1', 'R2']