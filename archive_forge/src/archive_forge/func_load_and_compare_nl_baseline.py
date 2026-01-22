import itertools
import re
from difflib import SequenceMatcher, unified_diff
from pyomo.repn.tests.diffutils import compare_floats, load_baseline
import pyomo.repn.plugins.nl_writer as nl_writer
def load_and_compare_nl_baseline(baseline, testfile, version='nl'):
    return nl_diff(*load_nl_baseline(baseline, testfile, version))