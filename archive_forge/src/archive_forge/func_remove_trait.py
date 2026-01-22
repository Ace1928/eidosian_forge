import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def remove_trait(self, session, trait, ignore_missing=True):
    """Remove a trait from the node.

        :param session: The session to use for making this request.
        :param trait: The trait to remove from the node.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the trait does not exist.
            Otherwise, ``False`` is returned.
        :returns bool: True on success removing the trait.
            False when the trait does not exist already.
        """
    session = self._get_session(session)
    version = utils.pick_microversion(session, '1.37')
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'traits', trait)
    response = session.delete(request.url, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    if ignore_missing and response.status_code == 400:
        session.log.debug('Trait %(trait)s was already removed from node %(node)s', {'trait': trait, 'node': self.id})
        return False
    msg = 'Failed to remove trait {trait} from bare metal node {node}'
    exceptions.raise_from_response(response, error_message=msg.format(node=self.id, trait=trait))
    if self.traits:
        self.traits = list(set(self.traits) - {trait})
    return True