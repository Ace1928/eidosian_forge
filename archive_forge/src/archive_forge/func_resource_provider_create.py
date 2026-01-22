import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_provider_create(self, name='', parent_provider_uuid=None):
    if not name:
        name = self.rand_name(name='', prefix=RP_PREFIX)
    to_exec = 'resource provider create ' + name
    if parent_provider_uuid is not None:
        to_exec += ' --parent-provider ' + parent_provider_uuid
    res = self.openstack(to_exec, use_json=True)

    def cleanup():
        try:
            self.resource_provider_delete(res['uuid'])
        except CommandException as exc:
            err_message = str(exc).lower()
            if 'no resource provider' not in err_message:
                raise
    self.addCleanup(cleanup)
    return res