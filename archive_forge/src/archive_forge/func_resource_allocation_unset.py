import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_allocation_unset(self, consumer_uuid, provider=None, resource_class=None, use_json=True, columns=()):
    cmd = 'resource provider allocation unset %s' % consumer_uuid
    if resource_class:
        cmd += ' ' + ' '.join(('--resource-class %s' % rc for rc in resource_class))
    if provider:
        if isinstance(provider, str):
            provider = [provider]
        cmd += ' ' + ' '.join(('--provider %s' % rp_uuid for rp_uuid in provider))
    cmd += ' '.join((' --column %s' % c for c in columns))
    result = self.openstack(cmd, use_json=use_json)

    def cleanup(uuid):
        try:
            self.openstack('resource provider allocation delete ' + uuid)
        except CommandException as exc:
            if 'not found' in str(exc).lower():
                pass
    self.addCleanup(cleanup, consumer_uuid)
    return result