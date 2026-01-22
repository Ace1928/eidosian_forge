import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def wait_for_reservation(self, session, timeout=None):
    """Wait for a lock on the node to be released.

        Bare metal nodes in ironic have a reservation lock that
        is used to represent that a conductor has locked the node
        while performing some sort of action, such as changing
        configuration as a result of a machine state change.

        This lock can occur during power syncronization, and prevents
        updates to objects attached to the node, such as ports.

        Note that nothing prevents a conductor from acquiring the lock again
        after this call returns, so it should be treated as best effort.

        Returns immediately if there is no reservation on the node.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param timeout: How much (in seconds) to wait for the lock to be
            released. The value of ``None`` (the default) means no timeout.

        :return: This :class:`Node` instance.
        """
    if self.reservation is None:
        return self
    for count in utils.iterate_timeout(timeout, 'Timeout waiting for the lock to be released on node %s' % self.id):
        self.fetch(session)
        if self.reservation is None:
            return self
        session.log.debug('Still waiting for the lock to be released on node %(node)s, currently locked by conductor %(host)s', {'node': self.id, 'host': self.reservation})