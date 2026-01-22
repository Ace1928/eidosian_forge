import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_quotas(self, collections: List[str]=None, quota_names: List[str]=None, quota_types: List[str]=None, short_descriptions: List[str]=None, dry_run: bool=False):
    """
        Describes one or more of your quotas.

        :param      collections: The group names of the quotas.
        :type       collections: ``list`` of ``str``

        :param      quota_names: The names of the quotas.
        :type       quota_names: ``list`` of ``str``

        :param      quota_types: The resource IDs if these are
        resource-specific quotas, global if they are not.
        :type       quota_types: ``list`` of ``str``

        :param      short_descriptions: The description of the quotas.
        :type       short_descriptions: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: A ``list`` of Product Type
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadQuotas'
    data = {'DryRun': dry_run, 'Filters': {}}
    if collections is not None:
        data['Filters'].update({'Collections': collections})
    if quota_names is not None:
        data['Filters'].update({'QuotaNames': quota_names})
    if quota_types is not None:
        data['Filters'].update({'QuotaTypes': quota_types})
    if short_descriptions is not None:
        data['Filters'].update({'ShortDescriptions': short_descriptions})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['QuotaTypes']
    return response.json()