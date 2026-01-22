import glob
import importlib
from os.path import basename, dirname, isfile, join
import torch
from torch._export.db.case import (
from . import *  # noqa: F403
def filter_examples_by_support_level(support_level: SupportLevel):
    return {key: val for key, val in all_examples().items() if val.support_level == support_level}