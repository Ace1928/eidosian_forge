import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
def test_signature_destroying_intermediate_decorator(self):

    def add_one_to_first_bad_decorator(f):
        """Bad because it doesn't wrap the f signature (clobbers it)"""

        def decorated(a, *args, **kwargs):
            return f(a + 1, *args, **kwargs)
        return decorated
    add_two_to_second = argmap(lambda b: b + 2, 1)

    @add_two_to_second
    @add_one_to_first_bad_decorator
    def add_one_and_two(a, b):
        return (a, b)
    assert add_one_and_two(5, 5) == (6, 7)