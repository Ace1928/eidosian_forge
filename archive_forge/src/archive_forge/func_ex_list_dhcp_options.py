import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def ex_list_dhcp_options(self, default: bool=None, dhcp_options_set_id: list=None, domaine_names: list=None, domaine_name_servers: list=None, ntp_servers: list=None, tag_keys: list=None, tag_values: list=None, tags: list=None, dry_run: bool=False):
    """
        Retrieves information about the content of one or more
        DHCP options sets.

        :param      default: SIf true, lists all default DHCP options set.
        If false, lists all non-default DHCP options set.
        :type       default: ``list`` of ``bool``

        :param      dhcp_options_set_id: The IDs of the DHCP options sets.
        :type       dhcp_options_set_id: ``list`` of ``str``

        :param      domaine_names: The domain names used for the DHCP
        options sets.
        :type       domaine_names: ``list`` of ``str``

        :param      domaine_name_servers: The domain name servers used for
        the DHCP options sets.
        :type       domaine_name_servers: ``list`` of ``str``

        :param      ntp_servers: The Network Time Protocol (NTP) servers used
        for the DHCP options sets.
        :type       ntp_servers: ``list`` of ``str``

        :param      tag_keys: The keys of the tags associated with the DHCP
        options sets.
        :type       ntp_servers: ``list`` of ``str``

        :param      tag_values: The values of the tags associated with the
        DHCP options sets.
        :type       tag_values: ``list`` of ``str``

        :param      tags: The key/value combination of the tags associated
        with the DHCP options sets, in the following format:
        "Filters":{"Tags":["TAGKEY=TAGVALUE"]}.
        :type       tags: ``list`` of ``str``

        :param      dry_run: If true, checks whether you have the required
        permissions to perform the action.
        :type       dry_run: ``bool``

        :return: a ``list`` of Dhcp Options
        :rtype: ``list`` of ``dict``
        """
    action = 'ReadDhcpOptions'
    data = {'DryRun': dry_run, 'Filters': {}}
    if default is not None:
        data['Filters'].update({'Default': default})
    if dhcp_options_set_id is not None:
        data['Filters'].update({'DhcpOptionsSetIds': dhcp_options_set_id})
    if domaine_names is not None:
        data['Filters'].update({'DomaineNames': domaine_names})
    if domaine_name_servers is not None:
        data['Filters'].update({'DomaineNameServers': domaine_name_servers})
    if ntp_servers is not None:
        data['Filters'].update({'NtpServers': ntp_servers})
    if tag_keys is not None:
        data['Filters'].update({'TagKeys': tag_keys})
    if tag_values is not None:
        data['Filters'].update({'TagValues': tag_values})
    if tags is not None:
        data['Filters'].update({'Tags': tags})
    response = self._call_api(action, json.dumps(data))
    if response.status_code == 200:
        return response.json()['DhcpOptionsSets']
    return response.json()