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
def step_5_remove_descriptors(self):
    """remove list of Properties from each compound (hardcoded)
        which would corrupt process of creating Prediction-Models"""
    sd_tags = ['activity__comment', 'alogp', 'assay__chemblid', 'assay__description', 'assay__type', 'bioactivity__type', 'activity_comment', 'assay_chemblid', 'assay_description', 'assay_type', 'bioactivity_type', 'cansmirdkit', 'ingredient__cmpd__chemblid', 'ingredient_cmpd_chemblid', 'knownDrug', 'medChemFriendly', 'molecularFormula', 'name__in__reference', 'name_in_reference', 'numRo5Violations', 'operator', 'organism', 'parent__cmpd__chemblid', 'parent_cmpd_chemblid', 'passesRuleOfThree', 'preferredCompoundName', 'reference', 'rotatableBonds', 'smiles', 'Smiles', 'stdInChiKey', 'synonyms', 'target__chemblid', 'target_chemblid', 'target__confidence', 'target__name', 'target_confidence', 'target_name', 'units', 'value_avg', 'value_stddev'] + ['value']
    result = []
    for mol in self.sd_entries:
        properties = mol.GetPropNames()
        for tag in properties:
            if tag in sd_tags:
                mol.ClearProp(tag)
        result.append(mol)
    self.sd_entries = result
    return True