from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import concatMap
def self_to_out_signature(func: FunctionSchema) -> FunctionSchema:
    assert func.kind() == SchemaKind.inplace
    assert func.arguments.self_arg is not None
    return FunctionSchema(name=func.name.remove_inplace().with_overload(get_expected_out_variant_overload_name(func.name.overload_name)), arguments=func.arguments.remove_self_annotation().with_out_args([Argument(name='out', type=func.arguments.self_arg.argument.type, default=None, annotation=func.arguments.self_arg.argument.annotation)]), returns=func.returns)