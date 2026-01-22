import argparse
import csv
import os
import sys
from rdkit import Chem
def existingFile(filename):
    """ 'type' for argparse - check that filename exists """
    if not os.path.exists(filename):
        raise argparse.ArgumentTypeError('{0} does not exist'.format(filename))
    return filename