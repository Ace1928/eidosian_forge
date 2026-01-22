from typing import cast
import pytest
import cirq
def skip_first(op):
    first = True
    for item in op:
        if not first:
            yield item
        first = False