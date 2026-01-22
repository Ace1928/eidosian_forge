import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter
import importlib
import memory_profiler as mp
def function_labels(dotted_function_names):
    state = {}

    def set_state_for(function_names, level):
        for fn in function_names:
            label = '.'.join(fn.split('.')[-level:])
            label_state = state.setdefault(label, {'functions': [], 'level': level})
            label_state['functions'].append(fn)
    set_state_for(dotted_function_names, 1)
    while True:
        ambiguous_labels = [label for label in state if len(state[label]['functions']) > 1]
        for ambiguous_label in ambiguous_labels:
            function_names = state[ambiguous_label]['functions']
            new_level = state[ambiguous_label]['level'] + 1
            del state[ambiguous_label]
            set_state_for(function_names, new_level)
        if len(ambiguous_labels) == 0:
            break
    fn_to_label = dict(((label_state['functions'][0], label) for label, label_state in state.items()))
    return fn_to_label