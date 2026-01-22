from pprint import pformat
from six import iteritems
import re
@container_name.setter
def container_name(self, container_name):
    """
        Sets the container_name of this V1ResourceFieldSelector.
        Container name: required for volumes, optional for env vars

        :param container_name: The container_name of this
        V1ResourceFieldSelector.
        :type: str
        """
    self._container_name = container_name