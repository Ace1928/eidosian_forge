from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
@with_native_function
def emit_decl_helper(g: NativeFunctionsViewGroup) -> Optional[str]:
    if g.view.has_composite_implicit_autograd_kernel:
        return None
    view_copy_inverse_sig = ViewInverseSignature(g)
    return view_copy_inverse_sig.decl()