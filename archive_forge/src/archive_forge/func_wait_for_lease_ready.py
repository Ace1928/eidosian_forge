import logging
from oslo_concurrency import lockutils
from oslo_context import context
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware import vim
from oslo_vmware import vim_util
def wait_for_lease_ready(self, lease):
    """Waits for the given lease to be ready.

        This method return when the lease is ready. In case of any error,
        appropriate exception is raised.

        :param lease: lease to be checked for
        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException
        """
    loop = loopingcall.FixedIntervalLoopingCall(self._poll_lease, lease)
    evt = loop.start(self._task_poll_interval)
    LOG.debug('Waiting for the lease: %s to be ready.', lease)
    evt.wait()