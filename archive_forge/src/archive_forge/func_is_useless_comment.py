import ast
import inspect
import textwrap
import warnings
import torch
def is_useless_comment(line):
    line = line.strip()
    return line.startswith('#') and (not line.startswith('# type:'))