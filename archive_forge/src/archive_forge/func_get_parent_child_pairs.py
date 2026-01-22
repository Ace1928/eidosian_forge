import inspect
import itertools
import textwrap
from typing import Callable, List, Mapping, Optional
import wandb
from .wandb_logging import wandb_log
import typing
from typing import NamedTuple
import collections
from collections import namedtuple
import kfp
from kfp import components
from kfp.components import InputPath, OutputPath
import wandb
def get_parent_child_pairs(full_func):
    components = full_func.split('.')
    parents, children = ([], [])
    for i, _ in enumerate(components[:-1], 1):
        parent = '.'.join(components[:i])
        child = components[i]
        parents.append(parent)
        children.append(child)
    return zip(parents, children)