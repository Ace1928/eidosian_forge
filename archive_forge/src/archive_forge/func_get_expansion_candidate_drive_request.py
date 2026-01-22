from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def get_expansion_candidate_drive_request():
    """Perform the request for expanding existing volume groups or disk pools.

            Note: the list of candidate structures do not necessarily produce candidates that meet all criteria.
            """
    candidates_list = None
    url = 'storage-systems/%s/symbol/getVolumeGroupExpansionCandidates?verboseErrorResponse=true' % self.ssid
    if self.raid_level == 'raidDiskPool':
        url = 'storage-systems/%s/symbol/getDiskPoolExpansionCandidates?verboseErrorResponse=true' % self.ssid
    try:
        rc, candidates_list = self.request(url, method='POST', data=self.pool_detail['id'])
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve volume candidates. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
    return candidates_list['candidates']