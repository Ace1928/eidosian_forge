from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..certificates import BoundCertificate
from ..core import BoundModelBase, ClientEntityBase, Meta
from ..load_balancer_types import BoundLoadBalancerType
from ..locations import BoundLocation
from ..metrics import Metrics
from ..networks import BoundNetwork
from ..servers import BoundServer
from .domain import (
def attach_to_network(self, load_balancer: LoadBalancer | BoundLoadBalancer, network: Network | BoundNetwork, ip: str | None=None) -> BoundAction:
    """Attach a Load Balancer to a Network.

        :param load_balancer: :class:` <hcloud.load_balancers.client.BoundLoadBalancer>` or :class:`LoadBalancer <hcloud.load_balancers.domain.LoadBalancer>`
        :param network: :class:`BoundNetwork <hcloud.networks.client.BoundNetwork>` or :class:`Network <hcloud.networks.domain.Network>`
        :param ip: str
                IP to request to be assigned to this Load Balancer
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    data: dict[str, Any] = {'network': network.id}
    if ip is not None:
        data.update({'ip': ip})
    response = self._client.request(url=f'/load_balancers/{load_balancer.id}/actions/attach_to_network', method='POST', json=data)
    return BoundAction(self._client.actions, response['action'])