from openstack import exceptions
from openstack import resource
from openstack import utils
def update_extra_specs_property(self, session, prop, val):
    """Update an extra spec for a flavor.

        :param session: The session to use for making this request.
        :param prop: The property to update.
        :param val: The value to update with.
        :returns: The updated value of the property.
        """
    url = utils.urljoin(Flavor.base_path, self.id, 'os-extra_specs', prop)
    microversion = self._get_microversion(session, action='commit')
    response = session.put(url, json={prop: val}, microversion=microversion)
    exceptions.raise_from_response(response)
    val = response.json().get(prop)
    return val