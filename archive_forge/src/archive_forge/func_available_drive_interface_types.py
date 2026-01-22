from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
@property
@memoize
def available_drive_interface_types(self):
    """Determine the types of available drives."""
    interfaces = [drive['phyDriveType'] for drive in self.drives]
    return [entry[0] for entry in get_most_common_elements(interfaces)]