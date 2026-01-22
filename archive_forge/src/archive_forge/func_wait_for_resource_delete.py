import os
import time
from cinderclient.v3 import client as cinderclient
import fixtures
from glanceclient import client as glanceclient
from keystoneauth1.exceptions import discovery as discovery_exc
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from keystoneclient import client as keystoneclient
from keystoneclient import discover as keystone_discover
from neutronclient.v2_0 import client as neutronclient
import openstack.config
import openstack.config.exceptions
from oslo_utils import uuidutils
import tempest.lib.cli.base
import testtools
import novaclient
import novaclient.api_versions
from novaclient import base
import novaclient.client
from novaclient.v2 import networks
import novaclient.v2.shell
def wait_for_resource_delete(self, resource, manager, timeout=60, poll_interval=1):
    """Wait until getting the resource raises NotFound exception.

        :param resource: Resource object.
        :param manager: Manager object with get method.
        :param timeout: timeout in seconds
        :param poll_interval: poll interval in seconds
        """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            manager.get(resource)
        except Exception as e:
            if getattr(e, 'http_status', None) == 404:
                break
            else:
                raise
        time.sleep(poll_interval)
    else:
        self.fail("The resource '%s' still exists." % base.getid(resource))