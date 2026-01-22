import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
def test_to_primitive_json_serializable(self):
    for in_val, _ in self.to_primitive_values:
        prim = self.field.to_primitive('obj', 'attr', in_val)
        jsencoded = jsonutils.dumps(prim)
        self.assertEqual(prim, jsonutils.loads(jsencoded))