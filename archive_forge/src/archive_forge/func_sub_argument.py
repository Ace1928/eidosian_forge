from typing import List, Optional, Sequence, Set, Union
from torchgen import local
from torchgen.api.types import (
from torchgen.model import (
from torchgen.utils import assert_never
from .types import (
def sub_argument(a: Union[Argument, TensorOptionsArguments, SelfArgument]) -> List[Binding]:
    return argument(a, cpp_no_default_args=cpp_no_default_args, method=method, faithful=faithful, has_tensor_options=has_tensor_options)