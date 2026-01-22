import copy
import fixtures
import uuid
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneauth1.tests.unit import k2k_fixtures
from testtools import matchers
from keystoneclient import access
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
from keystoneclient.v3.contrib.federation import base
from keystoneclient.v3.contrib.federation import identity_providers
from keystoneclient.v3.contrib.federation import mappings
from keystoneclient.v3.contrib.federation import protocols
from keystoneclient.v3.contrib.federation import service_providers
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
def build_parts(self, idp_id, protocol_id=None):
    """Build array used to construct mocking URL.

        Construct and return array with URL parts later used
        by methods like utils.TestCase.stub_entity().
        Example of URL:
        ``OS-FEDERATION/identity_providers/{idp_id}/
        protocols/{protocol_id}``

        """
    parts = ['OS-FEDERATION', 'identity_providers', idp_id, 'protocols']
    if protocol_id:
        parts.append(protocol_id)
    return parts