import argparse
import contextlib
import copy
import csv
import functools
import glob
import itertools
import logging
import math
import os
import tempfile
from collections import defaultdict, namedtuple
from dataclasses import replace
from typing import Any, Dict, Generator, Iterator, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils import benchmark
def pretty_plot(results, title, units: str, filename=None, dash_key='', legend_loc='lower right'):
    """Graph out the contents of a dict.
    Dash key means that if the result label has this key, then it will be displayed with a dash
    """
    if not filename:
        filename = title + '.png'
    filename = filename.replace(' ', '_').replace('/', '_').replace('-', '_').replace(':', '')
    workloads: Dict[str, Any] = {k: [] for v in results.values() for k in v.keys()}
    for v in results.values():
        for k in v.keys():
            workloads[k].append(float(v[k]))
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)
    for k, v in workloads.items():
        if dash_key and dash_key in k:
            plt.plot(list(results.keys()), v, '--')
        else:
            plt.plot(list(results.keys()), v)
    plt.title(title)
    plt.legend(list(workloads.keys()), loc=legend_loc)
    plt.ylabel(units)
    plt.xticks(rotation=45)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(f)