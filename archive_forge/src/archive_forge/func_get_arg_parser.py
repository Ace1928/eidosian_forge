import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def get_arg_parser():
    parser = argparse.ArgumentParser(description='Reads a Nearley grammar (with js functions), and outputs an equivalent lark parser.')
    parser.add_argument('nearley_grammar', help='Path to the file containing the nearley grammar')
    parser.add_argument('start_rule', help='Rule within the nearley grammar to make the base rule')
    parser.add_argument('nearley_lib', help='Path to root directory of nearley codebase (used for including builtins)')
    parser.add_argument('--es6', help='Enable experimental ES6 support', action='store_true')
    return parser