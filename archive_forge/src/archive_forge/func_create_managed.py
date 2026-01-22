from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple
from ..actions import ActionsPageResult, BoundAction, ResourceActionsClient
from ..core import BoundModelBase, ClientEntityBase, Meta
from .domain import (
def create_managed(self, name: str, domain_names: list[str], labels: dict[str, str] | None=None) -> CreateManagedCertificateResponse:
    """Creates a new managed Certificate with the given name and domain names. This methods allows only creating
           managed certificates for domains that are using the Hetzner DNS service. If you want to create a custom uploaded certificate use :func:`~hcloud.certificates.client.CertificatesClient.create`

        :param name: str
        :param domain_names: List[str]
               Domains and subdomains that should be contained in the Certificate
        :param labels: Dict[str, str] (optional)
               User-defined labels (key-value pairs)
        :return: :class:`BoundCertificate <hcloud.certificates.client.BoundCertificate>`
        """
    data: dict[str, Any] = {'name': name, 'type': Certificate.TYPE_MANAGED, 'domain_names': domain_names}
    if labels is not None:
        data['labels'] = labels
    response = self._client.request(url='/certificates', method='POST', json=data)
    return CreateManagedCertificateResponse(certificate=BoundCertificate(self, response['certificate']), action=BoundAction(self._client.actions, response['action']))