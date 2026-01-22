import inspect
import re
import six
def ui_command_set(self, group=None, **parameter):
    """
        Sets one or more configuration parameters in the given group.
        The "global" group contains all global CLI preferences.
        Other groups are specific to the current path.

        Run with no parameter nor group to list all available groups, or
        with just a group name to list all available parameters within that
        group.

        Example: set global color_mode=true loglevel_console=info

        SEE ALSO
        ========
        get
        """
    if group is None:
        self.shell.con.epy_write('\n                                     AVAILABLE CONFIGURATION GROUPS\n                                     ==============================\n                                     %s\n                                     ' % ' '.join(self.list_config_groups()))
    elif not parameter:
        if group not in self.list_config_groups():
            raise ExecutionError('Unknown configuration group: %s' % group)
        section = '%s CONFIG GROUP' % group.upper()
        underline1 = ''.ljust(len(section), '=')
        parameters = ''
        for p_name in self.list_group_params(group, writable=True):
            p_def = self.get_group_param(group, p_name)
            type_method = self.get_type_method(p_def['type'])
            p_name = '%s=%s' % (p_def['name'], p_def['type'])
            underline2 = ''.ljust(len(p_name), '-')
            parameters += '%s\n%s\n%s\n\n' % (p_name, underline2, p_def['description'])
        self.shell.con.epy_write('%s\n%s\n%s\n' % (section, underline1, parameters))
    elif group not in self.list_config_groups():
        raise ExecutionError('Unknown configuration group: %s' % group)
    for param, value in six.iteritems(parameter):
        if param not in self.list_group_params(group):
            raise ExecutionError("Unknown parameter %s in group '%s'." % (param, group))
        p_def = self.get_group_param(group, param)
        type_method = self.get_type_method(p_def['type'])
        if not p_def['writable']:
            raise ExecutionError('Parameter %s is read-only.' % param)
        try:
            value = type_method(value)
        except ValueError as msg:
            raise ExecutionError('Not setting %s! %s' % (param, msg))
        group_setter = self.get_group_setter(group)
        group_setter(param, value)
        group_getter = self.get_group_getter(group)
        value = group_getter(param)
        value = type_method(value, reverse=True)
        self.shell.con.display("Parameter %s is now '%s'." % (param, value))