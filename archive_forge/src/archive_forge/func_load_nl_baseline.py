import itertools
import re
from difflib import SequenceMatcher, unified_diff
from pyomo.repn.tests.diffutils import compare_floats, load_baseline
import pyomo.repn.plugins.nl_writer as nl_writer
def load_nl_baseline(baseline, testfile, version='nl'):
    return load_baseline(baseline, testfile, 'nl', version)