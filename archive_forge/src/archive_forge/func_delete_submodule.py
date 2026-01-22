import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer
from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode
import torch
from torch.nn import *
@compatibility(is_backward_compatible=True)
def delete_submodule(self, target: str) -> bool:
    """
        Deletes the given submodule from ``self``.

        The module will not be deleted if ``target`` is not a valid
        target.

        Args:
            target: The fully-qualified string name of the new submodule
                (See example in ``nn.Module.get_submodule`` for how to
                specify a fully-qualified string.)

        Returns:
            bool: Whether or not the target string referenced a
                submodule we want to delete. A return value of ``False``
                means that the ``target`` was not a valid reference to
                a submodule.
        """
    atoms = target.split('.')
    path, target_submod = (atoms[:-1], atoms[-1])
    mod: torch.nn.Module = self
    for item in path:
        if not hasattr(mod, item):
            return False
        mod = getattr(mod, item)
        if not isinstance(mod, torch.nn.Module):
            return False
    if not hasattr(mod, target_submod):
        return False
    if not isinstance(getattr(mod, target_submod), torch.nn.Module):
        return False
    delattr(mod, target_submod)
    return True