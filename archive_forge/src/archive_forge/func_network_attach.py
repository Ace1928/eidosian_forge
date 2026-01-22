from urllib import parse
from zunclient import api_versions
from zunclient.common import base
from zunclient.common import utils
from zunclient import exceptions
def network_attach(self, container, **kwargs):
    return self._action(container, '/network_attach', qparams=kwargs)