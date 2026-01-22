from pprint import pformat
from six import iteritems
import re
@config_source.setter
def config_source(self, config_source):
    """
        Sets the config_source of this V1NodeSpec.
        If specified, the source to get node configuration from The
        DynamicKubeletConfig feature gate must be enabled for the Kubelet to use
        this field

        :param config_source: The config_source of this V1NodeSpec.
        :type: V1NodeConfigSource
        """
    self._config_source = config_source