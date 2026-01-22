from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def is_user_in_group(self, name_or_id, group_name_or_id):
    """Check to see if a user is in a group.

        :param name_or_id: Name or unique ID of the user.
        :param group_name_or_id: Group name or ID

        :returns: True if user is in the group, False otherwise
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call
        """
    user, group = self._get_user_and_group(name_or_id, group_name_or_id)
    return self.identity.check_user_in_group(user, group)