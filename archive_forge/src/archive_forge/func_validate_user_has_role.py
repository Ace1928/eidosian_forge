from openstack.common import tag
from openstack import resource
from openstack import utils
def validate_user_has_role(self, session, user, role):
    """Validates that a user has a role on a project"""
    url = utils.urljoin(self.base_path, self.id, 'users', user.id, 'roles', role.id)
    resp = session.head(url)
    if resp.status_code == 204:
        return True
    return False