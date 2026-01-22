from __future__ import (absolute_import, division, print_function)
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel
def check_options(self, task, data):
    """
        Method for options validation to use in 'load_data' for TaskInclude and HandlerTaskInclude
        since they share the same validations. It is not named 'validate_options' on purpose
        to prevent confusion with '_validate_*" methods. Note that the task passed might be changed
        as a side-effect of this method.
        """
    my_arg_names = frozenset(task.args.keys())
    bad_opts = my_arg_names.difference(self.VALID_ARGS)
    if bad_opts and task.action in C._ACTION_ALL_PROPER_INCLUDE_IMPORT_TASKS:
        raise AnsibleParserError('Invalid options for %s: %s' % (task.action, ','.join(list(bad_opts))), obj=data)
    if not task.args.get('_raw_params'):
        task.args['_raw_params'] = task.args.pop('file', None)
        if not task.args['_raw_params']:
            raise AnsibleParserError('No file specified for %s' % task.action)
    apply_attrs = task.args.get('apply', {})
    if apply_attrs and task.action not in C._ACTION_INCLUDE_TASKS:
        raise AnsibleParserError('Invalid options for %s: apply' % task.action, obj=data)
    elif not isinstance(apply_attrs, dict):
        raise AnsibleParserError('Expected a dict for apply but got %s instead' % type(apply_attrs), obj=data)
    return task