import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
@classmethod
def find_subsystems(cls, context):
    """
        Find all subsystems in sysfs.

        :param Context context: the context
        :rtype: frozenset
        :returns: subsystems in sysfs
        """
    sys_path = context.sys_path
    dirnames = ('bus', 'class', 'subsystem')
    absnames = (os.path.join(sys_path, name) for name in dirnames)
    realnames = (d for d in absnames if os.path.isdir(d))
    return frozenset((n for d in realnames for n in os.listdir(d)))