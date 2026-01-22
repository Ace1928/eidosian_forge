import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
def test_to_primitive(self):
    for in_val, prim_val in self.to_primitive_values:
        self.assertEqual(prim_val, self.field.to_primitive('obj', 'attr', in_val))