import csv
import io
import logging
import os
import freewilson as fw
from rdkit import Chem, rdBase
def test_multicore():
    scaffolds = [Chem.MolFromSmiles('c1ccccc1[*].NC=O'), Chem.MolFromSmiles('C1CCCCC1')]
    mols = [Chem.MolFromSmiles(x) for x in ['c1ccccc1CC2CNC2C(=O)N', 'Cc1ccccc1CC2CNC2C(=O)N', 'Cc1ccccc1CC2CNCC(=O)NC2', 'C3c1ccccc1CC2CNC2C(=O)N3', 'C1CCCCC1F', 'ClC1CCCCC1F']]
    decomp = fw.FWDecompose(scaffolds, mols, [1, 2, 3, 4, 5, 6])
    s = io.StringIO()
    fw.predictions_to_csv(s, decomp, fw.FWBuild(decomp))