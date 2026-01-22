from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils import ls_helper
from gslib.utils import retry_util
from gslib.utils import text_util
from gslib.utils import unit_util
import gslib.tests.testcase as testcase
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TestParams
from gslib.utils.text_util import CompareVersions
from gslib.utils.unit_util import DecimalShort
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import PrettyTime
import httplib2
import os
import six
from six import add_move, MovedModule
from six.moves import mock
def testMakeBitsHumanReadable(self):
    """Tests converting bit counts to human-readable strings."""
    self.assertEqual(unit_util.MakeBitsHumanReadable(0), '0 bit')
    self.assertEqual(unit_util.MakeBitsHumanReadable(1023), '1023 bit')
    self.assertEqual(unit_util.MakeBitsHumanReadable(1024), '1 Kibit')
    self.assertEqual(unit_util.MakeBitsHumanReadable(1024 ** 2), '1 Mibit')
    self.assertEqual(unit_util.MakeBitsHumanReadable(1024 ** 3), '1 Gibit')
    self.assertEqual(unit_util.MakeBitsHumanReadable(1024 ** 3 * 5.3), '5.3 Gibit')
    self.assertEqual(unit_util.MakeBitsHumanReadable(1024 ** 4 * 2.7), '2.7 Tibit')
    self.assertEqual(unit_util.MakeBitsHumanReadable(1024 ** 5), '1 Pibit')
    self.assertEqual(unit_util.MakeBitsHumanReadable(1024 ** 6), '1 Eibit')