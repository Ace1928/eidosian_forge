from openstack import exceptions
from openstack.instance_ha.v1 import host as _host
from openstack.instance_ha.v1 import notification as _notification
from openstack.instance_ha.v1 import segment as _segment
from openstack.instance_ha.v1 import vmove as _vmove
from openstack import proxy
from openstack import resource
def delete_host(self, host, segment_id=None, ignore_missing=True):
    """Delete the host.

        :param segment_id: The ID of a failover segment.
        :param host: The value can be the ID of a host or a :class:
            `~masakariclient.sdk.ha.v1.host.Host` instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be
            raised when the host does not exist.
            When set to ``True``, no exception will be set when
            attempting to delete a nonexistent host.

        :returns: ``None``
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        :raises: :class:`~openstack.exceptions.InvalidRequest`
            when segment_id is None.

        """
    if segment_id is None:
        raise exceptions.InvalidRequest("'segment_id' must be specified.")
    host_id = resource.Resource._get_id(host)
    return self._delete(_host.Host, host_id, segment_id=segment_id, ignore_missing=ignore_missing)