import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def wait_for_provision_state(self, session, expected_state, timeout=None, abort_on_failed_state=True):
    """Wait for the node to reach the expected state.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param expected_state: The expected provisioning state to reach.
        :param timeout: If ``wait`` is set to ``True``, specifies how much (in
            seconds) to wait for the expected state to be reached. The value of
            ``None`` (the default) means no client-side timeout.
        :param abort_on_failed_state: If ``True`` (the default), abort waiting
            if the node reaches a failure state which does not match the
            expected one. Note that the failure state for ``enroll`` ->
            ``manageable`` transition is ``enroll`` again.

        :return: This :class:`Node` instance.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if the node
            reaches an error state and ``abort_on_failed_state`` is ``True``.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` on timeout.
        """
    for count in utils.iterate_timeout(timeout, "Timeout waiting for node %(node)s to reach target state '%(state)s'" % {'node': self.id, 'state': expected_state}):
        self.fetch(session)
        if self._check_state_reached(session, expected_state, abort_on_failed_state):
            return self
        session.log.debug('Still waiting for node %(node)s to reach state "%(target)s", the current state is "%(state)s"', {'node': self.id, 'target': expected_state, 'state': self.provision_state})