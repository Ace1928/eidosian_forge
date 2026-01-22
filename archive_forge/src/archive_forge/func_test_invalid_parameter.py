import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_invalid_parameter(self):

    def raises():

        @specs.parameter('x')
        def f(p):
            return p
    self.assertRaises(exceptions.NoParameterFoundException, raises)