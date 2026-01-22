from __future__ import print_function
import argparse
import os
import sys
from importlib import import_module
from jinja2 import Template
from palettable.palette import Palette
def find_palettes(mod):
    """
    Find all Palette instances in mod.

    """
    return {k: v for k, v in vars(mod).items() if isinstance(v, Palette) and (not k.endswith('_r'))}