from openstack import exceptions
from openstack import resource
from openstack import utils
def get_private_access(self, session):
    """List projects with private access to the volume type.

        :param session: The session to use for making this request.
        :returns: The volume type access response.
        """
    url = utils.urljoin(self.base_path, self.id, 'os-volume-type-access')
    resp = session.get(url)
    exceptions.raise_from_response(resp)
    return resp.json().get('volume_type_access', [])