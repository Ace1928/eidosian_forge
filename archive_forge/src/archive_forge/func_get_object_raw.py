import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def get_object_raw(self, container, obj, query_string=None, stream=False):
    """Get a raw response object for an object.

        :param string container: Name of the container.
        :param string obj: Name of the object.
        :param string query_string: Query args for uri. (delimiter, prefix,
            etc.)
        :param bool stream: Whether to stream the response or not.

        :returns: A `requests.Response`
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    endpoint = self._get_object_endpoint(container, obj, query_string)
    return self.object_store.get(endpoint, stream=stream)