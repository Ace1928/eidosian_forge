import urllib
from oslo_log import log as logging
from oslo_utils import timeutils
from glance.common import exception
from glance.i18n import _, _LE
def unpack_task_input(task):
    """Verifies and returns valid task input dictionary.

    :param task: Task domain object
    """
    task_type = task.type
    task_input = task.task_input
    if task_type == 'api_image_import':
        if not task_input:
            msg = _('Input to api_image_import task is empty.')
            raise exception.Invalid(msg)
        if 'image_id' not in task_input:
            msg = _("Missing required 'image_id' field")
            raise exception.Invalid(msg)
    else:
        for key in ['import_from', 'import_from_format', 'image_properties']:
            if key not in task_input:
                msg = _("Input does not contain '%(key)s' field") % {'key': key}
                raise exception.Invalid(msg)
    return task_input