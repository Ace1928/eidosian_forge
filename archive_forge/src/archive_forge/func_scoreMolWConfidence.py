import gzip
import math
import os.path
import pickle
import sys
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def scoreMolWConfidence(mol, fscore):
    """Next to the NP Likeness Score, this function outputs a confidence value
  between 0..1 that descibes how many fragments of the tested molecule
  were found in the model data set (1: all fragments were found).

  Returns namedtuple NPLikeness(nplikeness, confidence)"""
    if mol is None:
        raise ValueError('invalid molecule')
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    bits = fp.GetNonzeroElements()
    score = 0.0
    bits_found = 0
    for bit in bits:
        if bit in fscore:
            bits_found += 1
            score += fscore[bit]
    score /= float(mol.GetNumAtoms())
    confidence = float(bits_found / len(bits))
    if score > 4:
        score = 4.0 + math.log10(score - 4.0 + 1.0)
    elif score < -4:
        score = -4.0 - math.log10(-4.0 - score + 1.0)
    NPLikeness = namedtuple('NPLikeness', 'nplikeness,confidence')
    return NPLikeness(score, confidence)