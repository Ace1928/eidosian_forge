import inspect
import re
import six
def list_config_groups(self):
    """
        Lists the configuration group names.
        """
    return self._configuration_groups.keys()