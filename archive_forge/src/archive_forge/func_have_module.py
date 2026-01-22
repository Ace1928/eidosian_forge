import os
import sys
from subprocess import Popen
import importlib
from pathlib import Path
import warnings
import argparse
from multiprocessing import cpu_count
from ase.calculators.calculator import names as calc_names
from ase.cli.main import CLIError
def have_module(module):
    return importlib.find_loader(module) is not None