import abc
import itertools
from oslo_serialization import jsonutils
from neutron_lib import constants as const
from neutron_lib.db import constants as db_const
from neutron_lib.objects import common_types
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
def test_from_primitive(self):

    class ObjectLikeThing(object):
        _context = 'context'
    for prim_val, out_val in self.from_primitive_values:
        from_prim = self.field.from_primitive(ObjectLikeThing, 'attr', prim_val)
        self.assertEqual(out_val, from_prim)
        self.field.coerce('obj', 'attr', from_prim)