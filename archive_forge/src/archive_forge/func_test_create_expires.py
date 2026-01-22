import uuid
from oslo_utils import timeutils
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import application_credentials
def test_create_expires(self):
    ref = self.new_ref(user=uuid.uuid4().hex)
    ref['expires_at'] = timeutils.parse_isotime('2013-03-04T12:00:01.000000Z')
    req_ref = ref.copy()
    req_ref.pop('id')
    user = req_ref.pop('user')
    req_ref['expires_at'] = '2013-03-04T12:00:01.000000Z'
    self.stub_entity('POST', ['users', user, self.collection_key], status_code=201, entity=req_ref)
    super(ApplicationCredentialTests, self).test_create(ref=ref, req_ref=req_ref)