from builtins import zip
from builtins import str
import os
import os.path as op
import sys
from xml.etree import cElementTree as ET
import pyxnat
def subj_compare(item1, item2):
    """Compare sort of items"""
    return cmp(item1.label(), item2.label())