from openstack import _log
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
Wait for the node to reach the expected state.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param timeout: How much (in seconds) to wait for the introspection.
            The value of ``None`` (the default) means no client-side timeout.
        :param ignore_error: If ``True``, this call will raise an exception
            if the introspection reaches the ``error`` state. Otherwise the
            error state is considered successful and the call returns.
        :return: This :class:`Introspection` instance.
        :raises: :class:`~openstack.exceptions.ResourceFailure` if
            introspection fails and ``ignore_error`` is ``False``.
        :raises: :class:`~openstack.exceptions.ResourceTimeout` on timeout.
        