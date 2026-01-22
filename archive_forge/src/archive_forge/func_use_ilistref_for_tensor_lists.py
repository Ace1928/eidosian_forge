import threading
from contextlib import contextmanager
from typing import Iterator, Optional
def use_ilistref_for_tensor_lists() -> bool:
    assert _locals.use_ilistref_for_tensor_lists is not None, 'need to initialize local.use_ilistref_for_tensor_lists with local.parametrize'
    return _locals.use_ilistref_for_tensor_lists