from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient.v1 import databases
def change_passwords(self, instance, users):
    """Change the password for one or more users."""
    instance_id = base.getid(instance)
    user_dict = {'users': users}
    url = '/instances/%s/users' % instance_id
    resp, body = self.api.client.put(url, body=user_dict)
    common.check_for_exceptions(resp, body, url)