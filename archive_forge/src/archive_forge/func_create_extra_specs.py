from openstack import exceptions
from openstack import resource
from openstack import utils
def create_extra_specs(self, session, specs):
    """Creates extra specs for a flavor.

        :param session: The session to use for making this request.
        :param specs:
        :returns: The updated flavor.
        """
    url = utils.urljoin(Flavor.base_path, self.id, 'os-extra_specs')
    microversion = self._get_microversion(session, action='create')
    response = session.post(url, json={'extra_specs': specs}, microversion=microversion)
    exceptions.raise_from_response(response)
    specs = response.json().get('extra_specs', {})
    self._update(extra_specs=specs)
    return self