import re
from difflib import SequenceMatcher, unified_diff
from pyomo.repn.tests.diffutils import compare_floats, load_baseline
def load_lp_baseline(baseline, testfile, version='lp'):
    return load_baseline(baseline, testfile, 'lp', version)