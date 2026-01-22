from pprint import pformat
from six import iteritems
import re
@finished_at.setter
def finished_at(self, finished_at):
    """
        Sets the finished_at of this V1ContainerStateTerminated.
        Time at which the container last terminated

        :param finished_at: The finished_at of this V1ContainerStateTerminated.
        :type: datetime
        """
    self._finished_at = finished_at