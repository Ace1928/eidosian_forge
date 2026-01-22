import collections
import contextlib
from tensorflow.python import pywrap_tfe
@contextlib.contextmanager
def push_forwardprop_state():
    """Temporarily push or pop transient state for accumulators in the active set.

  Allows an accumulator which is currently processing an operation to
  temporarily reset its state. This is useful when building forwardprop versions
  of functions, where an accumulator will trigger function building and then
  must process captured symbolic tensors while building it. Without pushing and
  popping, accumulators ignore operations executed as a direct result of their
  own jvp computations.

  Yields:
    None (used for its side effect).
  """
    try:
        pywrap_tfe.TFE_Py_ForwardAccumulatorPushState()
        yield
    finally:
        pywrap_tfe.TFE_Py_ForwardAccumulatorPopState()