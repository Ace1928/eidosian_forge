import dataclasses
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import torch
from torch._export import ExportedProgram
from torch.utils._pytree import (
def default_from_dumpable_context(dumpable_context: DumpableContext) -> Context:
    return (SERIALIZED_DATACLASS_TO_PYTHON_DATACLASS[dumpable_context[0]], dumpable_context[1], dumpable_context[2])