from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def get_member_definitions(group, include_failed=False):
    """Get member definitions in (name, ResourceDefinition) pair for group.

    The List is sorted first by created_time then by name.
    If include_failed is set, failed members will be put first in the
    List sorted by created_time then by name.
    """
    inspector = GroupInspector.from_parent_resource(group)
    template = inspector.template()
    if template is None:
        return []
    definitions = template.resource_definitions(None)
    return [(name, definitions[name]) for name in inspector.member_names(include_failed=include_failed) if name in definitions]