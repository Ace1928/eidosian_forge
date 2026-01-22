from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_deserialize_entity_null_base(self):
    deser_ent = self.ser_null.deserialize_entity('context', 'entity')
    self.assertEqual('entity', deser_ent)