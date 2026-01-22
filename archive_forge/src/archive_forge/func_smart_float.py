import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def smart_float(number):
    number_as_string = '%s' % float(number)
    if 'e' in number_as_string:
        return '%.4f' % float(number)
    else:
        return number_as_string