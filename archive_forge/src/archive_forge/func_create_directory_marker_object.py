import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def create_directory_marker_object(self, container, name, **headers):
    """Create a zero-byte directory marker object

        .. note::

          This method is not needed in most cases. Modern swift does not
          require directory marker objects. However, some swift installs may
          need these.

        When using swift Static Web and Web Listings to serve static content
        one may need to create a zero-byte object to represent each
        "directory". Doing so allows Web Listings to generate an index of the
        objects inside of it, and allows Static Web to render index.html
        "files" that are "inside" the directory.

        :param container: The name of the container.
        :param name: Name for the directory marker object within the container.
        :param headers: These will be passed through to the object creation
            API as HTTP Headers.
        :returns: The created object store ``Object`` object.
        """
    headers['content-type'] = 'application/directory'
    return self.create_object(container, name, data='', generate_checksums=False, **headers)