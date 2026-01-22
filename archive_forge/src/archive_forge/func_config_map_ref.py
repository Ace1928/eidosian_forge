from pprint import pformat
from six import iteritems
import re
@config_map_ref.setter
def config_map_ref(self, config_map_ref):
    """
        Sets the config_map_ref of this V1EnvFromSource.
        The ConfigMap to select from

        :param config_map_ref: The config_map_ref of this V1EnvFromSource.
        :type: V1ConfigMapEnvSource
        """
    self._config_map_ref = config_map_ref