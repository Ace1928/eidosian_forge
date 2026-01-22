from __future__ import absolute_import, division, print_function
def delete_roleassignment(self, assignment_id):
    """
        Deletes specified role assignment.

        :return: True
        """
    self.log('Deleting the role assignment {0}'.format(self.name))
    try:
        response = self.authorization_client.role_assignments.delete_by_id(role_id=assignment_id)
    except Exception as e:
        self.log('Error attempting to delete the role assignment.')
        self.fail('Error deleting the role assignment: {0}'.format(str(e)))
    return True