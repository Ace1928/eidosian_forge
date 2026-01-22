import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def revoke_project_role_from_user(self, project, role, user):
    """
        Revoke project role from a user.

        :param project: Project to revoke the role from.
        :type project: :class:`.OpenStackIdentityDomain`

        :param role: Role to revoke.
        :type role: :class:`.OpenStackIdentityRole`

        :param user: User to revoke the role from.
        :type user: :class:`.OpenStackIdentityUser`

        :return: ``True`` on success.
        :rtype: ``bool``
        """
    path = '/v3/projects/{}/users/{}/roles/{}'.format(project.id, user.id, role.id)
    response = self.authenticated_request(path, method='DELETE')
    return response.status == httplib.NO_CONTENT