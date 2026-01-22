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
def patch_kfp():
    to_patch = [('kfp.components', create_component_from_func), ('kfp.components._python_op', create_component_from_func), ('kfp.components._python_op', _get_function_source_definition), ('kfp.components._python_op', strip_type_hints)]
    successes = []
    for module_name, func in to_patch:
        success = patch(module_name, func)
        successes.append(success)
    if not all(successes):
        wandb.termerror('Failed to patch one or more kfp functions.  Patching @wandb_log decorator to no-op.')
        patch('wandb.integration.kfp', wandb_log)