import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import regions
def test_update_enabled_defaults_to_none(self):
    super(RegionTests, self).test_update(req_ref={'description': uuid.uuid4().hex})