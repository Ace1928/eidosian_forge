import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def test_overload_operators_in_subthread():
    """Test we can create a model in a child thread with overloaded operators."""
    worker1 = threading.Thread(target=_overload_plus, args=('+', 0))
    worker2 = threading.Thread(target=_overload_plus, args=('*', 1))
    worker2.start()
    worker1.start()
    worker1.join()
    worker2.join()
    worker1 = threading.Thread(target=_overload_plus, args=('+', 1))
    worker2 = threading.Thread(target=_overload_plus, args=('*', 0))
    worker2.start()
    worker1.start()
    worker1.join()
    worker2.join()