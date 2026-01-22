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
def step_6_calc_descriptors(self):
    """calculate descriptors for each compound, according to Descriptors._descList"""
    nms = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
    for i in range(len(self.sd_entries)):
        descrs = calc.CalcDescriptors(self.sd_entries[i])
        for j in range(len(descrs)):
            self.sd_entries[i].SetProp(str(nms[j]), str(descrs[j]))
    return True