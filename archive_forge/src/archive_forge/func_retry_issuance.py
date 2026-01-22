from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import (
def retry_issuance(self, certificate: Certificate | BoundCertificate) -> BoundAction:
    """Returns all action objects for a Certificate.

        :param certificate: :class:`BoundCertificate <hcloud.certificates.client.BoundCertificate>` or :class:`Certificate <hcloud.certificates.domain.Certificate>`
        :return: :class:`BoundAction <hcloud.actions.client.BoundAction>`
        """
    response = self._client.request(url=f'/certificates/{certificate.id}/actions/retry', method='POST')
    return BoundAction(self._client.actions, response['action'])