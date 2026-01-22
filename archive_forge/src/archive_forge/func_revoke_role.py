from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def revoke_role(self, name_or_id, user=None, group=None, project=None, domain=None, system=None, wait=False, timeout=60):
    """Revoke a role from a user.

        :param string name_or_id: Name or unique ID of the role.
        :param string user: The name or id of the user.
        :param string group: The name or id of the group. (v3)
        :param string project: The name or id of the project.
        :param string domain: The id of the domain. (v3)
        :param bool system: The name of the system. (v3)
        :param bool wait: Wait for role to be revoked
        :param int timeout: Timeout to wait for role to be revoked

            NOTE: for wait and timeout, sometimes revoking roles is not
            instantaneous.

            NOTE: project is required for keystone v2

        NOTE: precedence is given first to project, then domain, then system

        :returns: True if the role is revoke, otherwise False
        :raises: :class:`~openstack.exceptions.SDKException` if the role cannot
            be removed
        """
    data = self._get_grant_revoke_params(name_or_id, user=user, group=group, project=project, domain=domain, system=system)
    user = data.get('user')
    group = data.get('group')
    project = data.get('project')
    domain = data.get('domain')
    role = data.get('role')
    if project:
        if user:
            has_role = self.identity.validate_user_has_project_role(project, user, role)
            if not has_role:
                self.log.debug('Assignment does not exists')
                return False
            self.identity.unassign_project_role_from_user(project, user, role)
        else:
            has_role = self.identity.validate_group_has_project_role(project, group, role)
            if not has_role:
                self.log.debug('Assignment does not exists')
                return False
            self.identity.unassign_project_role_from_group(project, group, role)
    elif domain:
        if user:
            has_role = self.identity.validate_user_has_domain_role(domain, user, role)
            if not has_role:
                self.log.debug('Assignment does not exists')
                return False
            self.identity.unassign_domain_role_from_user(domain, user, role)
        else:
            has_role = self.identity.validate_group_has_domain_role(domain, group, role)
            if not has_role:
                self.log.debug('Assignment does not exists')
                return False
            self.identity.unassign_domain_role_from_group(domain, group, role)
    elif user:
        has_role = self.identity.validate_user_has_system_role(user, role, system)
        if not has_role:
            self.log.debug('Assignment does not exist')
            return False
        self.identity.unassign_system_role_from_user(user, role, system)
    else:
        has_role = self.identity.validate_group_has_system_role(group, role, system)
        if not has_role:
            self.log.debug('Assignment does not exist')
            return False
        self.identity.unassign_system_role_from_group(group, role, system)
    return True