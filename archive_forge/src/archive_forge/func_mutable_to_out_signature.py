from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import concatMap
def mutable_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    assert func.kind() == SchemaKind.mutable
    new_returns, new_out_args = generate_out_args_from_schema(func)
    return FunctionSchema(name=func.name.remove_inplace().with_overload(get_expected_out_variant_overload_name(func.name.overload_name)), arguments=func.arguments.with_out_args(new_out_args), returns=tuple(new_returns))