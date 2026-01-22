from pprint import pformat
from six import iteritems
import re
@env_from.setter
def env_from(self, env_from):
    """
        Sets the env_from of this V1alpha1PodPresetSpec.
        EnvFrom defines the collection of EnvFromSource to inject into
        containers.

        :param env_from: The env_from of this V1alpha1PodPresetSpec.
        :type: list[V1EnvFromSource]
        """
    self._env_from = env_from