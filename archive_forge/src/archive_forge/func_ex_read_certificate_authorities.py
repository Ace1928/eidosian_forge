import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_read_certificate_authorities(self, ca_fingerprints: List[str]=None, ca_ids: List[str]=None, descriptions: List[str]=None, dry_run: bool=False):
    """
        Returns information about one or more of your Client Certificate
        Authorities (CAs).

        :param      ca_fingerprints: The fingerprints of the CAs.
        :type       ca_fingerprints: ``list`` of ``str``

        :param      ca_ids: The IDs of the CAs.
        :type       ca_ids: ``list`` of ``str``

        :param      descriptions: The descriptions of the CAs.
        :type       descriptions: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a list of all Ca matching filled filters.
        :rtype: ``list`` of  ``dict``
        """
    action = 'ReadCas'
    data = {'DryRun': dry_run, 'Filters': {}}
    if ca_fingerprints is not None:
        data['Filters'].update({'CaFingerprints': ca_fingerprints})
    if ca_ids is not None:
        data['Filters'].update({'CaIds': ca_ids})
    if descriptions is not None:
        data['Filters'].update({'Descriptions': descriptions})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['Cas']
    return response.json()