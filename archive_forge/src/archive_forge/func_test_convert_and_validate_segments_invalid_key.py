from webob import exc
from neutron_lib.api.definitions import provider_net
from neutron_lib.api.validators import multiprovidernet as mp_validator
from neutron_lib import constants
from neutron_lib.tests import _base as base
def test_convert_and_validate_segments_invalid_key(self):
    segs = [_build_segment(seg_id=2)]
    segs[0]['some_key'] = 'some_value'
    self.assertRaises(exc.HTTPBadRequest, mp_validator.convert_and_validate_segments, segs)