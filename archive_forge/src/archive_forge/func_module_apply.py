import os
from troveclient import base
from troveclient import common
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules as core_modules
from swiftclient import client as swift_client
def module_apply(self, instance, modules):
    """Apply modules to an instance."""
    url = '/instances/%s/modules' % base.getid(instance)
    body = {'modules': self._get_module_list(modules)}
    resp, body = self.api.client.post(url, body=body)
    common.check_for_exceptions(resp, body, url)
    return [core_modules.Module(self, module, loaded=True) for module in body['modules']]