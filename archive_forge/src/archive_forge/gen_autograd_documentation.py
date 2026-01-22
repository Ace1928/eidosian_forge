import argparse
import os
from typing import List
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.gen import parse_native_yaml
from torchgen.selective_build.selector import SelectiveBuilder
from . import gen_python_functions
from .gen_autograd_functions import (
from .gen_inplace_or_view_type import gen_inplace_or_view_type
from .gen_trace_type import gen_trace_type
from .gen_variable_factories import gen_variable_factories
from .gen_variable_type import gen_variable_type
from .load_derivatives import load_derivatives

To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.autograd.gen_autograd        aten/src/ATen/native/native_functions.yaml        aten/src/ATen/native/tags.yaml        $OUTPUT_DIR        tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/autograd/generated/
