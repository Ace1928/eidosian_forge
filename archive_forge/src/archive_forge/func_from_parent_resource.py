from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
@classmethod
def from_parent_resource(cls, parent_resource):
    """Create a GroupInspector from a parent resource.

        This is a convenience method to instantiate a GroupInspector from a
        Heat StackResource object.
        """
    return cls(parent_resource.context, parent_resource.rpc_client(), parent_resource.nested_identifier())