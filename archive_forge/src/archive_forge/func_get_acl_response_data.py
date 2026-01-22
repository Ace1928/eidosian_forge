from oslo_utils import timeutils
import requests_mock
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
def get_acl_response_data(self, operation_type='read', users=None, project_access=False):
    if users is None:
        users = self.users1
    op_data = {'users': users}
    op_data['project-access'] = project_access
    op_data['created'] = self.created
    op_data['updated'] = str(timeutils.utcnow())
    acl_data = {operation_type: op_data}
    return acl_data