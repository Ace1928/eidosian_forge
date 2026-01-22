import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_default_repetitions():

    class MyResult(cirq.Result):

        def __init__(self, records):
            self._records = records

        @property
        def params(self):
            raise NotImplementedError()

        @property
        def measurements(self):
            raise NotImplementedError()

        @property
        def records(self):
            return self._records

        @property
        def data(self):
            raise NotImplementedError()
    assert MyResult({}).repetitions == 0
    assert MyResult({'a': np.zeros((5, 2, 3))}).repetitions == 5