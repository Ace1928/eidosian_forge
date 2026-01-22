import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def is_anaconda():
    import glob
    conda_pattern = os.path.join(sys.prefix, 'conda-meta\\graphviz*.json')
    return glob.glob(conda_pattern) != []