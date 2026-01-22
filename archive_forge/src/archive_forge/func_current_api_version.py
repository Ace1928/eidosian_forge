import logging
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient.common.http import DEFAULT_VER
from ironicclient.v1 import allocation
from ironicclient.v1 import chassis
from ironicclient.v1 import conductor
from ironicclient.v1 import deploy_template
from ironicclient.v1 import driver
from ironicclient.v1 import events
from ironicclient.v1 import node
from ironicclient.v1 import port
from ironicclient.v1 import portgroup
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
@property
def current_api_version(self):
    """Return the current API version in use.

        This returns the version of the REST API that the API client
        is presently set to request. This value may change as a result
        of API version negotiation.
        """
    return self.http_client.os_ironic_api_version