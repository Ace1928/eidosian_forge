import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_product_types(self, product_type_ids: List[str]=None, dry_run: bool=False):
    """
        Describes one or more product types.

        :param      product_type_ids: The IDs of the product types.
        :type       product_type_ids: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: A ``list`` of Product Type
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadProductTypes'
    data = {'DryRun': dry_run, 'Filters': {}}
    if product_type_ids is not None:
        data['Filters'].update({'ProductTypeIds': product_type_ids})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['ProductTypes']
    return response.json()