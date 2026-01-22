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
def choose_how_many_workers(jobs):
    if jobs == MULTIPROCESSING_AUTO:
        if have_module('xdist'):
            jobs = min(cpu_count(), MULTIPROCESSING_MAX_WORKERS)
        else:
            jobs = MULTIPROCESSING_DISABLED
    return jobs