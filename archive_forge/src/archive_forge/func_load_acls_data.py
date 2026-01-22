import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
def load_acls_data(self):
    """Loads ACL entity from Barbican server using its acl_ref

        Clears the existing list of per operation ACL settings if there.
        Populates current ACL entity with ACL settings received from Barbican
        server.

        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
    response = self._api.get(self.acl_ref_relative)
    del self.operation_acls[:]
    for op_type in response:
        acl_dict = response.get(op_type)
        proj_access = acl_dict.get('project-access')
        users = acl_dict.get('users')
        created = acl_dict.get('created')
        updated = acl_dict.get('updated')
        self.add_operation_acl(operation_type=op_type, project_access=proj_access, users=users, created=created, updated=updated)