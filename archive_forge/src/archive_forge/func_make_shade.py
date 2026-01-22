import sys
import pbr.version
from os_client_config import cloud_config
from os_client_config.config import OpenStackConfig  # noqa
from os_client_config import vendors  # noqa
def make_shade(options=None, **kwargs):
    """Simple wrapper for getting a Shade OpenStackCloud object

    A mechanism that matches make_sdk, make_client and make_rest_client.

    :rtype: :class:`~shade.OpenStackCloud`
    """
    import shade
    cloud = get_config(options=options, **kwargs)
    return shade.OpenStackCloud(cloud_config=cloud, **kwargs)