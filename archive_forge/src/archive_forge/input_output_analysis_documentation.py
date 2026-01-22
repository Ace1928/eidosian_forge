import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from .functional_utils import _get_mutation_type
from .schemas import (
from .utils import strict_zip

This module is one of the analysis modules - it takes as input a function or graph
and some preexisting properties, and returns some data that is useful for deciding
how to further proceed with compilation or construct runtime wrappers.

In particular, the following analyses are provided:
1. Refine the view and mutation metadata collected previously - removing duplicate
   inputs or mapping views to their bases.
2. We also analyze the function signature for export graphs.
