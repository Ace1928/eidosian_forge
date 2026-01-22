from pprint import pformat
from six import iteritems
import re
@updated_number_scheduled.setter
def updated_number_scheduled(self, updated_number_scheduled):
    """
        Sets the updated_number_scheduled of this V1beta2DaemonSetStatus.
        The total number of nodes that are running updated daemon pod

        :param updated_number_scheduled: The updated_number_scheduled of this
        V1beta2DaemonSetStatus.
        :type: int
        """
    self._updated_number_scheduled = updated_number_scheduled