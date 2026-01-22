from the base class `SymbolDoc`, and put the extra doc as the docstring
import re as _re
from .base import build_param_doc as _build_param_doc
@staticmethod
def get_output_shape(sym, **input_shapes):
    """Get user friendly information of the output shapes."""
    _, s_outputs, _ = sym.infer_shape(**input_shapes)
    return dict(zip(sym.list_outputs(), s_outputs))