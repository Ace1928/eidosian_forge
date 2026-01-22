import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
def test_coerce_good_values(self):
    for in_val, out_val in self.coerce_good_values:
        self.assertEqual(out_val, self.field.coerce('obj', 'attr', in_val))