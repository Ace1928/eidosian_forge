from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def validate_assignment_properties(self):
    if self.properties.get(self.ROLES) is not None:
        for role_assignment in self.properties.get(self.ROLES):
            project = role_assignment.get(self.PROJECT)
            domain = role_assignment.get(self.DOMAIN)
            if project is not None and domain is not None:
                raise exception.ResourcePropertyConflict(self.PROJECT, self.DOMAIN)
            if project is None and domain is None:
                msg = _('Either project or domain must be specified for role %s') % role_assignment.get(self.ROLE)
                raise exception.StackValidationFailed(message=msg)