import builtins
import copy
import dataclasses
import inspect
import io
import math
import pathlib
import sys
import typing
from enum import auto, Enum
from typing import (
import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager
from torch.utils._pytree import (
from .exported_program import ExportedProgram, ModuleCallEntry, ModuleCallSignature
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
@property
def serializable_spec(self):
    return {'t_id': self.t_id, 'dim': self.dim, 'min': self.constraint_range.vr.lower, 'max': self.constraint_range.vr.upper, 'shared': None if self.shared is None else {'t_id': self.shared.t_id, 'dim': self.shared.dim}}