from unittest import mock
from oslo_config import cfg
from webob import exc
from neutron_lib.api.validators import allowedaddresspairs as validator
from neutron_lib.exceptions import allowedaddresspairs as addr_exc
from neutron_lib.tests import _base as base
def test__validate_allowed_address_pairs_not_a_list(self):
    for d in [{}, set(), 'abc', True, 1]:
        self.assertRaisesRegex(exc.HTTPBadRequest, 'must be a list', validator._validate_allowed_address_pairs, d)