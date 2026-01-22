import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite

        Verify that a patch matching the self axis with a predicate doesn't
        cause an infinite loop. See <http://genshi.edgewall.org/ticket/82>.
        