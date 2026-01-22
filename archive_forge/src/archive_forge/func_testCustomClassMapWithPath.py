import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
def testCustomClassMapWithPath(self):

    class ExampleClass(Mapping[Any, Any]):
        """Small example custom class."""

        def __init__(self, *args, **kwargs):
            self._mapping = dict(*args, **kwargs)

        def __getitem__(self, k: Any) -> Any:
            return self._mapping[k]

        def __len__(self) -> int:
            return len(self._mapping)

        def __iter__(self) -> Iterator[Any]:
            return iter(self._mapping)

    def mapper(path, value):
        full_path = '/'.join(path)
        return f'{full_path}_{value}'
    test_input = ExampleClass({'first': 1, 'nested': {'second': 2, 'third': 3}})
    output = tree.map_structure_with_path(mapper, test_input)
    expected = ExampleClass({'first': 'first_1', 'nested': {'second': 'nested/second_2', 'third': 'nested/third_3'}})
    self.assertEqual(output, expected)