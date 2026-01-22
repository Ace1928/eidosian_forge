import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_slugify(self):
    to_slug = strutils.to_slug
    self.assertRaises(TypeError, to_slug, True)
    self.assertEqual('hello', to_slug('hello'))
    self.assertEqual('two-words', to_slug('Two Words'))
    self.assertEqual('ma-any-spa-ce-es', to_slug('Ma-any\t spa--ce- es'))
    self.assertEqual('excamation', to_slug('exc!amation!'))
    self.assertEqual('ampserand', to_slug('&ampser$and'))
    self.assertEqual('ju5tnum8er', to_slug('ju5tnum8er'))
    self.assertEqual('strip-', to_slug(' strip - '))
    self.assertEqual('perche', to_slug('perchÃ©'.encode('latin-1')))
    self.assertEqual('strange', to_slug('\x80strange', errors='ignore'))