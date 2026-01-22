from pprint import pformat
from six import iteritems
import re
@exit_code.setter
def exit_code(self, exit_code):
    """
        Sets the exit_code of this V1ContainerStateTerminated.
        Exit status from the last termination of the container

        :param exit_code: The exit_code of this V1ContainerStateTerminated.
        :type: int
        """
    if exit_code is None:
        raise ValueError('Invalid value for `exit_code`, must not be `None`')
    self._exit_code = exit_code