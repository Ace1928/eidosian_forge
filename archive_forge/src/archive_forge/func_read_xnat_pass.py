import os
from functools import partial
def read_xnat_pass(f):
    if os.path.exists(f) and os.path.isfile(f):
        infile = open(f)
        return parse_xnat_pass(infile.readlines())
    else:
        return None