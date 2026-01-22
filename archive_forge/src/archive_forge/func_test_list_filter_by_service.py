import uuid
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import registered_limits
def test_list_filter_by_service(self):
    service_id = uuid.uuid4().hex
    expected_query = {'service_id': service_id}
    self.test_list(expected_query=expected_query, service=service_id)