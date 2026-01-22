import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_allocation_set(self, consumer_uuid, allocations, project_id=None, user_id=None, consumer_type=None, use_json=True, may_print_to_stderr=False):
    cmd = 'resource provider allocation set {allocs} {uuid}'.format(uuid=consumer_uuid, allocs=' '.join(('--allocation {}'.format(a) for a in allocations)))
    if project_id:
        cmd += ' --project-id %s' % project_id
    if user_id:
        cmd += ' --user-id %s' % user_id
    if consumer_type:
        cmd += ' --consumer-type %s' % consumer_type
    result = self.openstack(cmd, use_json=use_json, may_print_to_stderr=may_print_to_stderr)

    def cleanup(uuid):
        try:
            self.openstack('resource provider allocation delete ' + uuid)
        except CommandException as exc:
            if 'not found' in str(exc).lower():
                pass
    self.addCleanup(cleanup, consumer_uuid)
    return result