from __future__ import print_function
import sys
import os
import argparse
from .unparser import roundtrip
from . import dump
def roundtrip_recursive(target, dump_tree=False):
    if os.path.isfile(target):
        print(target)
        print('=' * len(target))
        if dump_tree:
            dump(target)
        else:
            roundtrip(target)
        print()
    elif os.path.isdir(target):
        for item in os.listdir(target):
            if item.endswith('.py'):
                roundtrip_recursive(os.path.join(target, item), dump_tree)
    else:
        print("WARNING: skipping '%s', not a file or directory" % target, file=sys.stderr)