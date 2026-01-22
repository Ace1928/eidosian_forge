import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_nets_access_points(self, net_access_point_ids: List[str]=None, net_ids: List[str]=None, service_names: List[str]=None, states: List[str]=None, tag_keys: List[str]=None, tag_values: List[str]=None, tags: List[str]=None, dry_run: bool=False):
    """
        Describes one or more Net access points.

        :param      net_access_point_ids: The IDs of the Net access points.
        :type       net_access_point_ids: ``list`` of ``str``

        :param      net_ids: The IDs of the Nets.
        :type       net_ids: ``list`` of ``str``

        :param      service_names: The The names of the prefix lists
        corresponding to the services. For more information,
        see DescribePrefixLists:
        https://docs.outscale.com/api#describeprefixlists
        :type       service_names: ``list`` of ``str``

        :param      states: The states of the Net access points
        (pending | available | deleting | deleted).
        :type       states: ``list`` of ``str``

        :param      tag_keys: The keys of the tags associated with the Net
        access points.
        :type       tag_keys: ``list`` of ``str``

        :param      tag_values: The values of the tags associated with the
        Net access points.
        :type       tag_values: ``list`` of ``str``

        :param      tags: The key/value combination of the tags associated
        with the Net access points, in the following format:
        "Filters":{"Tags":["TAGKEY=TAGVALUE"]}.
        :type       tags: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: A list of Net Access Points
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadNetAccessPoints'
    data = {'DryRun': dry_run, 'Filters': {}}
    if net_access_point_ids is not None:
        data['Filters'].update({'NetAccessPointIds': net_access_point_ids})
    if net_ids is not None:
        data['Filters'].update({'NetIds': net_ids})
    if service_names is not None:
        data['Filters'].update({'ServiceNames': service_names})
    if states is not None:
        data['Filters'].update({'States': states})
    if tag_keys is not None:
        data['Filters'].update({'TagKeys': tag_keys})
    if tag_values is not None:
        data['Filters'].update({'TagValues': tag_values})
    if tags is not None:
        data['Filters'].update({'Tags': tags})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['NetAccessPoints']
    return response.json()