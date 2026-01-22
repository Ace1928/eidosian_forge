from openstack.common import tag
from openstack import resource
from openstack import utils
def assign_role_to_user(self, session, user, role):
    """Assign role to user on project"""
    url = utils.urljoin(self.base_path, self.id, 'users', user.id, 'roles', role.id)
    resp = session.put(url)
    if resp.status_code == 204:
        return True
    return False