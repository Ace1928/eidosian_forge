from __future__ import (absolute_import, division, print_function)
import ast
from pathlib import PosixPath
import yaml
import argparse
import os
def is_extending_collection(result, col_fqcn):
    if result:
        for x in result.get('extends_documentation_fragment', []):
            if x.startswith(col_fqcn):
                return True
    return False