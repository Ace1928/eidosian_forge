from openstack import exceptions
from openstack.instance_ha.v1 import host as _host
from openstack.instance_ha.v1 import notification as _notification
from openstack.instance_ha.v1 import segment as _segment
from openstack.instance_ha.v1 import vmove as _vmove
from openstack import proxy
from openstack import resource
def get_vmove(self, vmove, notification):
    """Get a single vmove.

        :param vmove: The value can be the UUID of one vmove or
            a :class: `~masakariclient.sdk.ha.v1.vmove.VMove` instance.
        :param notification: The value can be the UUID of a notification or
            a :class: `~masakariclient.sdk.ha.v1.notification.Notification`
            instance.
        :returns: one 'VMove' resource class.
        :raises: :class:`~openstack.exceptions.ResourceNotFound`
            when no resource can be found.
        :raises: :class:`~openstack.exceptions.InvalidRequest`
            when notification_id is None.
        """
    notification_id = resource.Resource._get_id(notification)
    vmove_id = resource.Resource._get_id(vmove)
    return self._get(_vmove.VMove, vmove_id, notification_id=notification_id)