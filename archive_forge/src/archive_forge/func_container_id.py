from pprint import pformat
from six import iteritems
import re
@container_id.setter
def container_id(self, container_id):
    """
        Sets the container_id of this V1ContainerStateTerminated.
        Container's ID in the format 'docker://<container_id>'

        :param container_id: The container_id of this
        V1ContainerStateTerminated.
        :type: str
        """
    self._container_id = container_id