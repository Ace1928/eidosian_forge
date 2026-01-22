from __future__ import absolute_import, division, print_function
def list_assignments(self, filter=None):
    """
        Returns a list of assignments.
        """
    results = []
    try:
        response = list(self.authorization_client.role_assignments.list(filter=filter))
        response = [self.roleassignment_to_dict(a) for a in response]
        if self.role_definition_id:
            response = [role_assignment for role_assignment in response if role_assignment.get('role_definition_id').split('/')[-1].lower() == self.role_definition_id.split('/')[-1].lower()]
        results = response
    except Exception as ex:
        self.log("Didn't find role assignments in subscription {0}.".format(self.subscription_id))
    return results