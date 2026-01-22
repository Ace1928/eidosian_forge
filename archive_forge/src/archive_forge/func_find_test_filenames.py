from __future__ import print_function
import sys
import os
import types
import traceback
import pprint
import argparse
from srsly.ruamel_yaml.compat import PY3
def find_test_filenames(directory):
    filenames = {}
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            base, ext = os.path.splitext(filename)
            if base.endswith('-py2' if PY3 else '-py3'):
                continue
            filenames.setdefault(base, []).append(ext)
    filenames = sorted(filenames.items())
    return filenames