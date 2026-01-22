import inspect
import re
import six
def get_group_param(self, group, param):
    """
        @param group: The configuration group to retreive the parameter from.
        @type group: str
        @param param: The parameter name.
        @type param: str
        @return: A dictionnary for the requested group parameter, with
        name, writable, description, group and type fields.
        @rtype: dict
        @raise ValueError: If the parameter or group does not exist.
        """
    if group not in self.list_config_groups():
        raise ValueError('Not such configuration group %s' % group)
    if param not in self.list_group_params(group):
        raise ValueError('Not such parameter %s in configuration group %s' % (param, group))
    p_type, p_description, p_writable = self._configuration_groups[group][param]
    return dict(name=param, group=group, type=p_type, description=p_description, writable=p_writable)