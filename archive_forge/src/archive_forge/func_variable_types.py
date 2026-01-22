import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
def variable_types(self):
    vartypes = {}
    for t, vars in zip(('z', 'e', 'p'), self.variables()):
        for v in vars:
            vartypes[v] = t
    return vartypes