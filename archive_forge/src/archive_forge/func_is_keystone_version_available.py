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
def is_keystone_version_available(session, version):
    """Given a (major, minor) pair, check if the API version is enabled."""
    d = keystone_discover.Discover(session)
    try:
        d.create_client(version)
    except (discovery_exc.DiscoveryFailure, discovery_exc.VersionNotAvailable):
        return False
    else:
        return True