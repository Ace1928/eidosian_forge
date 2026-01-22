from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def delete_assignment(self, user_id=None, group_id=None):
    if self.properties[self.ROLES] is not None:
        current_assignments = self.parse_list_assignments(user_id=user_id, group_id=group_id)
        if user_id is not None:
            self._remove_role_assignments_from_user(user_id, self.properties[self.ROLES], current_assignments)
        elif group_id is not None:
            self._remove_role_assignments_from_group(group_id, self.properties[self.ROLES], current_assignments)