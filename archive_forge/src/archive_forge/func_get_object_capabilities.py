import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def get_object_capabilities(self):
    """Get infomation about the object-storage service

        The object-storage service publishes a set of capabilities that
        include metadata about maximum values and thresholds.

        :returns: An object store ``Info`` object.
        """
    return self.object_store.get_info()