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
def predictions_to_csv(outstream, decomposition: FreeWilsonDecomposition, predictions):
    """Output predictions in csv format to the output stream

       :param outstream: output stream to write results
       :param decomposition: freewillson decomposition
       :param predictions: list of Predictions to output
    """
    writer = None
    for pred in predictions:
        if not writer:
            rgroups = set()
            for rgroup in decomposition.rgroups:
                rgroups.add(rgroup)
            rgroups = sorted(rgroups, key=_rgroup_sort)
            lookup = {}
            for i, rg in enumerate(rgroups):
                lookup[rg] = i
            writer = csv.writer(outstream)
            header = ['smiles', 'prediction'] + [f'{rg}_smiles' for rg in list(rgroups)]
            writer.writerow(header)
        rg = [''] * len(lookup)
        for s in pred.rgroups:
            rg[lookup[s.rgroup]] = s.smiles
        row = [pred.smiles, repr(pred.prediction)] + rg
        writer.writerow(row)
    return header