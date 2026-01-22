from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_on_each_iterable_qid():

    class QidIter(cirq.Qid):

        @property
        def dimension(self) -> int:
            return 2

        def _comparison_key(self) -> Any:
            return 1

        def __iter__(self):
            raise NotImplementedError()
    assert cirq.H.on_each(QidIter())[0] == cirq.H.on(QidIter())