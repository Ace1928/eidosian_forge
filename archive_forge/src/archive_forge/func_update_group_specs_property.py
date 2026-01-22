from openstack import exceptions
from openstack import resource
from openstack import utils
def update_group_specs_property(self, session, prop, val):
    """Update a group spec property of the group type.

        :param session: The session to use for making this request.
        :param prop: The name of the group spec property to update.
        :param val: The value to set for the group spec property.
        :returns: The updated value of the group spec property.
        """
    url = utils.urljoin(GroupType.base_path, self.id, 'group_specs', prop)
    microversion = self._get_microversion(session, action='commit')
    response = session.put(url, json={prop: val}, microversion=microversion)
    exceptions.raise_from_response(response)
    val = response.json().get(prop)
    return val