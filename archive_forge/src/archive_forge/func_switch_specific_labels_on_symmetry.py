import re
import sys
from optparse import OptionParser
from rdkit import Chem
def switch_specific_labels_on_symmetry(smi, symmetry_class, a, b):
    if symmetry_class[a - 1] == symmetry_class[b - 1]:
        matchObj = re.search('\\[\\*\\:([123])\\].*\\[\\*\\:([123])\\].*\\[\\*\\:([123])\\]', smi)
        if matchObj:
            if int(matchObj.group(a)) > int(matchObj.group(b)):
                smi = re.sub('\\[\\*\\:' + matchObj.group(a) + '\\]', '[*:XX' + matchObj.group(b) + 'XX]', smi)
                smi = re.sub('\\[\\*\\:' + matchObj.group(b) + '\\]', '[*:XX' + matchObj.group(a) + 'XX]', smi)
                smi = re.sub('XX', '', smi)
    return smi