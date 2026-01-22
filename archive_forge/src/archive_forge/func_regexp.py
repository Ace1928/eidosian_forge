import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def regexp(self, r):
    return '/%s/' % r