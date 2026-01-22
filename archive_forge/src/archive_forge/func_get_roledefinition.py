from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils._text import to_native
def get_roledefinition(self):
    """
        Gets the properties of the specified role definition.

        :return: deserialized role definition state dictionary
        """
    self.log('Checking if the role definition {0} is present'.format(self.name))
    response = None
    try:
        response = list(self._client.role_definitions.list(scope=self.scope))
        if len(response) > 0:
            self.log('Response : {0}'.format(response))
            roles = []
            for r in response:
                if r.role_name == self.name:
                    roles.append(r)
            if len(roles) == 1:
                self.log('role definition : {0} found'.format(self.name))
                return roledefinition_to_dict(roles[0])
            if len(roles) > 1:
                self.fail('Found multiple role definitions: {0}'.format(roles))
    except Exception as ex:
        self.log("Didn't find role definition {0}".format(self.name))
    return False