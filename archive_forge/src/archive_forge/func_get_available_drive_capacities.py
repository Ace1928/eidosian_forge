from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_available_drive_capacities(self, drive_id_list=None):
    """Determine the list of available drive capacities."""
    if drive_id_list:
        available_drive_capacities = set([int(drive['usableCapacity']) for drive in self.drives if drive['id'] in drive_id_list and drive['available'] and (drive['status'] == 'optimal')])
    else:
        available_drive_capacities = set([int(drive['usableCapacity']) for drive in self.drives if drive['available'] and drive['status'] == 'optimal'])
    self.module.log('available drive capacities: %s' % available_drive_capacities)
    return list(available_drive_capacities)