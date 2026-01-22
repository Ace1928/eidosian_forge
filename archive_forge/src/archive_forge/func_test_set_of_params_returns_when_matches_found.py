import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_set_of_params_returns_when_matches_found(self):

    def func(apple, apricot, banana, carrot):
        pass
    argspec = inspection.ArgSpec(*inspect.getfullargspec(func))
    funcspec = inspection.FuncProps('func', argspec, False)
    com = autocomplete.ParameterNameCompletion()
    self.assertSetEqual(com.matches(1, 'a', funcprops=funcspec), {'apple=', 'apricot='})
    self.assertSetEqual(com.matches(2, 'ba', funcprops=funcspec), {'banana='})
    self.assertSetEqual(com.matches(3, 'car', funcprops=funcspec), {'carrot='})