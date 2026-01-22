import inspect
import re
import six
def list_group_params(self, group, writable=None):
    """
        Lists the parameters from group matching the optional param, writable
        and type supplied (if none is supplied, returns all group parameters.
        @param group: The group to list parameters of.
        @type group: str
        @param writable: Optional writable flag filter.
        @type writable: bool
        """
    if group not in self.list_config_groups():
        return []
    else:
        params = []
        for p_name, p_def in six.iteritems(self._configuration_groups[group]):
            p_type, p_description, p_writable = p_def
            if writable is not None and p_writable != writable:
                continue
            params.append(p_name)
        params.sort()
        return params