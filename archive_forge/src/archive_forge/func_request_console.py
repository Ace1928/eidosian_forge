from __future__ import annotations
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..datacenters import BoundDatacenter
from ..firewalls import BoundFirewall
from ..floating_ips import BoundFloatingIP
from ..images import BoundImage, CreateImageResponse
from ..isos import BoundIso
from ..metrics import Metrics
from ..placement_groups import BoundPlacementGroup
from ..primary_ips import BoundPrimaryIP
from ..server_types import BoundServerType
from ..volumes import BoundVolume
from .domain import (
def request_console(self, server: Server | BoundServer) -> RequestConsoleResponse:
    """Requests credentials for remote access via vnc over websocket to keyboard, monitor, and mouse for a server.

        :param server: :class:`BoundServer <hcloud.servers.client.BoundServer>` or :class:`Server <hcloud.servers.domain.Server>`
        :return: :class:`RequestConsoleResponse <hcloud.servers.domain.RequestConsoleResponse>`
        """
    response = self._client.request(url=f'/servers/{server.id}/actions/request_console', method='POST')
    return RequestConsoleResponse(action=BoundAction(self._client.actions, response['action']), wss_url=response['wss_url'], password=response['password'])