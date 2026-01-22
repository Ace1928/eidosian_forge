import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import (
from torchgen.model import (
from torchgen.utils import FileManager, mapMaybe
from .context import with_native_function_with_differentiability_info_and_key
from .gen_inplace_or_view_type import (
from .gen_trace_type import (
def emit_history() -> str:
    fn = 'rebase' if modifies_arguments(f) and view_info is None else 'set'
    output_names = [r.name for r in differentiable_outputs]
    outs = CodeTemplate('flatten_tensor_args( ${outs} )').substitute(outs=output_names if not is_inplace_foreach else 'self')
    if not is_inplace_foreach:
        return SET_HISTORY.substitute(fn=fn, differentiable_outputs=outs)
    else:
        return LOOP_OVER_VECTOR_OF_GRAD_FNS.substitute(preamble=f'auto differentiable_outputs = {outs};\nTORCH_INTERNAL_ASSERT(differentiable_outputs.size() == grad_fns.size());', statements=f'{fn}_history(differentiable_outputs[i], grad_fns[i]);')