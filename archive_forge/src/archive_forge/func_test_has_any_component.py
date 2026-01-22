import unittest
import aniso8601
from aniso8601.duration import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.tests.compat import mock
def test_has_any_component(self):
    self.assertTrue(_has_any_component('P1Y', ['Y', 'M']))
    self.assertFalse(_has_any_component('P1Y', ['M', 'D']))