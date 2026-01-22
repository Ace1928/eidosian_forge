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
def negotiate_api_version(self):
    """Triggers negotiation with the remote API endpoint.

        :returns: the negotiated API version.
        """
    return self.http_client.negotiate_version(self.http_client.session, None)