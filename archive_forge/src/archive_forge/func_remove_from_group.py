from debtcollector import renames
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def remove_from_group(self, user, group):
    """Remove the specified user from the specified group.

        :param user: the user to be removed from the group.
        :type user: str or :class:`keystoneclient.v3.users.User`
        :param group: the group to remove the user from.
        :type group: str or :class:`keystoneclient.v3.groups.Group`

        :returns: Response object with 204 status.
        :rtype: :class:`requests.models.Response`

        """
    self._require_user_and_group(user, group)
    base_url = '/groups/%s' % base.getid(group)
    return super(UserManager, self).delete(base_url=base_url, user_id=base.getid(user))