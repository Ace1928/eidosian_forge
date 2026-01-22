import json
from troveclient import base
from troveclient import common
def parameters_by_version(self, version):
    """Get a list of valid parameters that can be changed."""
    return self._list('/datastores/versions/%s/parameters' % version, 'configuration-parameters')