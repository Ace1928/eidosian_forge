import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def dispatch_trace_input(arg: Union[Argument, TensorOptionsArguments]) -> Sequence[str]:
    if isinstance(arg, TensorOptionsArguments):
        name = 'options'
        return [ADD_TRACE_INPUT.substitute(name=name, input='optTypeMetaToScalarType(options.dtype_opt())'), ADD_TRACE_INPUT.substitute(name=name, input='options.layout()'), ADD_TRACE_INPUT.substitute(name=name, input='options.device()'), ADD_TRACE_INPUT.substitute(name=name, input='options.pinned_memory()')]
    else:
        name = arg.name
        if str(arg.type) == 'Tensor?[]':
            return [f'jit::tracer::addInputs(node, "{name}", {name});']
        else:
            return [ADD_TRACE_INPUT.substitute(name=name, input=name)]