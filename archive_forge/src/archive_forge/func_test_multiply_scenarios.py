import unittest
import testtools
from testtools.matchers import EndsWith
from testtools.tests.helpers import LoggingResult
import testscenarios
from testscenarios.scenarios import (
def test_multiply_scenarios(self):

    def factory(name):
        for i in 'ab':
            yield (i, {name: i})
    scenarios = multiply_scenarios(factory('p'), factory('q'))
    self.assertEqual([('a,a', dict(p='a', q='a')), ('a,b', dict(p='a', q='b')), ('b,a', dict(p='b', q='a')), ('b,b', dict(p='b', q='b'))], scenarios)