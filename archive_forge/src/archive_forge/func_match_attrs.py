import collections
import copy
import itertools
import random
import re
import warnings
def match_attrs(elem):
    orig_clades = elem.__dict__.pop('clades')
    found = elem.find_any(target, **kwargs)
    elem.clades = orig_clades
    return found is not None