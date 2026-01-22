import csv
import gzip
import json
import math
import optparse
import os
import pickle
import re
import sys
from pickle import Unpickler
import numpy as np
import requests
from pylab import *
from scipy import interp, stats
from sklearn import cross_validation, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, make_scorer, precision_score, recall_score,
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SDWriter
from rdkit.ML.Descriptors import MoleculeDescriptors
from the one dimensional weights.
def save_mols(self, outfile, gzip=True):
    """create SD-File of current molecules in self.sd_entries"""
    sdw = Chem.SDWriter(outfile + '.tmp')
    for mol in self.sd_entries:
        sdw.write(mol)
    sdw.flush()
    sdw.close()
    if not gzip:
        os.rename(outfile + '.tmp', outfile)
        return
    f_in = open(outfile + '.tmp', 'rb')
    f_out = gzip.open(outfile, 'wb')
    f_out.writelines(f_in)
    f_out.flush()
    f_out.close()
    f_in.close()
    os.remove(outfile + '.tmp')
    return