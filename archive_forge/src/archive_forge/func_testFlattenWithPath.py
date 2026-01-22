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
@parameterized.parameters([dict(inputs=[], expected=[]), dict(inputs=[23, '42'], expected=[((0,), 23), ((1,), '42')]), dict(inputs=[[[[108]]]], expected=[((0, 0, 0, 0), 108)]), dict(inputs=Foo(a=3, b=Bar(c=23, d=42)), expected=[(('a',), 3), (('b', 'c'), 23), (('b', 'd'), 42)]), dict(inputs=Foo(a=Bar(c=23, d=42), b=Bar(c=0, d='thing')), expected=[(('a', 'c'), 23), (('a', 'd'), 42), (('b', 'c'), 0), (('b', 'd'), 'thing')]), dict(inputs=Bar(c=42, d=43), expected=[(('c',), 42), (('d',), 43)]), dict(inputs=Bar(c=[42], d=43), expected=[(('c', 0), 42), (('d',), 43)]), dict(inputs=wrapt.ObjectProxy(Bar(c=[42], d=43)), expected=[(('c', 0), 42), (('d',), 43)])])
def testFlattenWithPath(self, inputs, expected):
    self.assertEqual(tree.flatten_with_path(inputs), expected)