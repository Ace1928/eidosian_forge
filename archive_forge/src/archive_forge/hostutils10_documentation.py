import re
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import hostutils
from oslo_log import log as logging
Get host's assignable PCI devices.

        :returns: a list of the assignable PCI devices.
        