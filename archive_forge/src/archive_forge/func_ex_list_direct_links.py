import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_direct_links(self, direct_link_ids: list=None, dry_run: bool=False):
    """
        Lists all DirectLinks in the Region.

        :param      direct_link_ids: The IDs of the DirectLinks. (required)
        :type       direct_link_ids: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: ``list`` of  Direct Links
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadDirectLinks'
    data = {'DryRun': dry_run, 'Filters': {}}
    if direct_link_ids is not None:
        data['Filters'].update({'DirectLinkIds': direct_link_ids})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['DirectLinks']
    return response.json()