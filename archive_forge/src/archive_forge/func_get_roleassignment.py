from __future__ import absolute_import, division, print_function
def get_roleassignment(self):
    """
        Gets the properties of the specified role assignment.

        :return: deserialized role assignment dictionary
        """
    self.log('Checking if the role assignment {0} is present'.format(self.name))
    role_assignment = None
    if self.id:
        try:
            response = self.authorization_client.role_assignments.get_by_id(role_id=self.id)
            role_assignment = self.roleassignment_to_dict(response)
            if role_assignment and self.assignee_object_id and (role_assignment.get('assignee_object_id') != self.assignee_object_id):
                self.fail('State Mismatch Error: The assignment ID exists, but does not match the provided assignee.')
            if role_assignment and self.role_definition_id and (role_assignment.get('role_definition_id').split('/')[-1].lower() != self.role_definition_id.split('/')[-1].lower()):
                self.fail('State Mismatch Error: The assignment ID exists, but does not match the provided role.')
        except Exception as ex:
            self.log("Didn't find role assignments id {0}".format(self.id))
    elif self.name and self.scope:
        try:
            response = self.authorization_client.role_assignments.get(scope=self.scope, role_assignment_name=self.name)
            role_assignment = self.roleassignment_to_dict(response)
            if role_assignment and self.assignee_object_id and (role_assignment.get('assignee_object_id') != self.assignee_object_id):
                self.fail('State Mismatch Error: The assignment name exists, but does not match the provided assignee.')
            if role_assignment and self.role_definition_id and (role_assignment.get('role_definition_id').split('/')[-1].lower() != self.role_definition_id.split('/')[-1].lower()):
                self.fail('State Mismatch Error: The assignment name exists, but does not match the provided role.')
        except Exception as ex:
            self.log("Didn't find role assignment by name {0} at scope {1}".format(self.name, self.scope))
    else:
        try:
            if self.scope and self.assignee_object_id and self.role_definition_id:
                response = list(self.authorization_client.role_assignments.list())
                response = [self.roleassignment_to_dict(role_assignment) for role_assignment in response]
                response = [role_assignment for role_assignment in response if role_assignment.get('scope') == self.scope]
                response = [role_assignment for role_assignment in response if role_assignment.get('assignee_object_id') == self.assignee_object_id]
                response = [role_assignment for role_assignment in response if role_assignment.get('role_definition_id').split('/')[-1].lower() == self.role_definition_id.split('/')[-1].lower()]
            else:
                self.fail('If id or name are not supplied, then assignee_object_id and role_definition_id are required.')
            if response:
                role_assignment = response[0]
        except Exception as ex:
            self.log("Didn't find role assignments for subscription {0}".format(self.subscription_id))
    return role_assignment