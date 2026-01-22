import json
from troveclient import base
from troveclient import common
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
def list_all_parameter_by_version(self, version):
    """List all configuration parameters deleted or not."""
    return self._list('/mgmt/datastores/versions/%s/parameters' % version, 'configuration-parameters')