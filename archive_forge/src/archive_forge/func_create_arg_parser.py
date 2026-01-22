import ast
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
from nodedump import debug_format_node
def create_arg_parser(desc):
    parser = ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose')
    parser.add_argument('source', type=str, help='Python source')
    return parser