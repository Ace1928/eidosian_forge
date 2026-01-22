from __future__ import print_function
import argparse
import sys
import re
import numpy as np
import caffe_parser
import mxnet as mx
from convert_symbol import convert_symbol
def prob_label(arg_names):
    candidates = [arg for arg in arg_names if not arg.endswith('data') and (not arg.endswith('_weight')) and (not arg.endswith('_bias')) and (not arg.endswith('_gamma')) and (not arg.endswith('_beta'))]
    if len(candidates) == 0:
        return 'prob_label'
    return candidates[-1]