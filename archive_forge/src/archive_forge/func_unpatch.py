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
def unpatch(module_name):
    if module_name in wandb.patched:
        for module, func in wandb.patched[module_name]:
            setattr(module, func, getattr(module, f'orig_{func}'))
        wandb.patched[module_name] = []