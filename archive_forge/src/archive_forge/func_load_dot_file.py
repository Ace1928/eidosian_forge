import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def load_dot_file(filename):
    with open(filename, 'r') as f:
        dotdata = f.readlines()
    log.info('Data read from %s' % filename)
    return dotdata