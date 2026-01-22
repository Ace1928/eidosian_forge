from pprint import pformat
from six import iteritems
import re
@machine_id.setter
def machine_id(self, machine_id):
    """
        Sets the machine_id of this V1NodeSystemInfo.
        MachineID reported by the node. For unique machine identification in the
        cluster this field is preferred. Learn more from man(5) machine-id:
        http://man7.org/linux/man-pages/man5/machine-id.5.html

        :param machine_id: The machine_id of this V1NodeSystemInfo.
        :type: str
        """
    if machine_id is None:
        raise ValueError('Invalid value for `machine_id`, must not be `None`')
    self._machine_id = machine_id