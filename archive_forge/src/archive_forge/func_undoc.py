from typing import Sequence
from IPython.utils.docs import GENERATING_DOCUMENTATION
def undoc(func):
    """Mark a function or class as undocumented.

    This is found by inspecting the AST, so for now it must be used directly
    as @undoc, not as e.g. @decorators.undoc
    """
    return func