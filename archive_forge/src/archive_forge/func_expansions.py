import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def expansions(self, *x):
    return '%s' % '\n    |'.join(x)