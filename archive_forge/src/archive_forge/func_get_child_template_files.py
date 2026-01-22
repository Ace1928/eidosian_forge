from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def get_child_template_files(context, stack, is_rolling_update, old_template_id):
    """Return a merged map of old and new template files.

    For rolling update files for old and new defintions are required as the
    nested stack is updated in batches of scaled units.
    """
    if not stack.convergence:
        old_template_id = stack.t.id
    if is_rolling_update and old_template_id:
        prev_files = template.Template.load(context, old_template_id).files
        prev_files.update(dict(stack.t.files))
        return prev_files
    return stack.t.files