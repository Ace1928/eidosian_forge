import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def rule(self, name):
    return _get_rulename(name)