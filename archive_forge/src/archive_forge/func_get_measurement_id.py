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
def get_measurement_id(r):
    return (r[0].get(META_ALGORITHM, '').partition('@')[0], r[1].task_spec.label, r[1].task_spec.sub_label, r[1].task_spec.env)