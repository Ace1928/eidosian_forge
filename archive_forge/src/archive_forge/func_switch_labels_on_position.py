import re
import sys
from optparse import OptionParser
from rdkit import Chem
def switch_labels_on_position(smi):
    smi = re.sub('\\[\\*\\:[123]\\]', '[*:XX1XX]', smi, 1)
    smi = re.sub('\\[\\*\\:[123]\\]', '[*:XX2XX]', smi, 1)
    smi = re.sub('\\[\\*\\:[123]\\]', '[*:XX3XX]', smi, 1)
    smi = re.sub('XX', '', smi)
    return smi