import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def revoke_domain_role_from_user(self, domain, user, role):
    """
        Revoke domain role from a user.

        :param domain: Domain to revoke the role from.
        :type domain: :class:`.OpenStackIdentityDomain`

        :param role: Role to revoke.
        :type role: :class:`.OpenStackIdentityRole`

        :param user: User to revoke the role from.
        :type user: :class:`.OpenStackIdentityUser`

        :return: ``True`` on success.
        :rtype: ``bool``
        """
    path = '/v3/domains/{}/users/{}/roles/{}'.format(domain.id, user.id, role.id)
    response = self.authenticated_request(path, method='DELETE')
    return response.status == httplib.NO_CONTENT