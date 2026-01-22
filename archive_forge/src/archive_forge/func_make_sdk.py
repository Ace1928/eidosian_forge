import sys
import pbr.version
from os_client_config import cloud_config
from os_client_config.config import OpenStackConfig  # noqa
from os_client_config import vendors  # noqa
def make_sdk(options=None, **kwargs):
    """Simple wrapper for getting an OpenStack SDK Connection.

    For completeness, provide a mechanism that matches make_client and
    make_rest_client. The heavy lifting here is done in openstacksdk.

    :rtype: :class:`~openstack.connection.Connection`
    """
    from openstack import connection
    cloud = get_config(options=options, **kwargs)
    return connection.from_config(cloud_config=cloud, options=options)