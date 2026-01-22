import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def attach_vif(self, session, vif_id, retry_on_conflict=True):
    """Attach a VIF to the node.

        The exact form of the VIF ID depends on the network interface used by
        the node. In the most common case it is a Network service port
        (NOT a Bare Metal port) ID. A VIF can only be attached to one node
        at a time.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param string vif_id: Backend-specific VIF ID.
        :param retry_on_conflict: Whether to retry HTTP CONFLICT errors.
            This can happen when either the VIF is already used on a node or
            the node is locked. Since the latter happens more often, the
            default value is True.
        :return: ``None``
        :raises: :exc:`~openstack.exceptions.NotSupported` if the server
            does not support the VIF API.
        """
    session = self._get_session(session)
    version = self._assert_microversion_for(session, 'commit', _common.VIF_VERSION, error_message='Cannot use VIF attachment API')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'vifs')
    body = {'id': vif_id}
    retriable_status_codes = _common.RETRIABLE_STATUS_CODES
    if not retry_on_conflict:
        retriable_status_codes = list(set(retriable_status_codes) - {409})
    response = session.post(request.url, json=body, headers=request.headers, microversion=version, retriable_status_codes=retriable_status_codes)
    msg = 'Failed to attach VIF {vif} to bare metal node {node}'.format(node=self.id, vif=vif_id)
    exceptions.raise_from_response(response, error_message=msg)