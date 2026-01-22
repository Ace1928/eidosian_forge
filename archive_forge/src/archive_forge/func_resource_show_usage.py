import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_show_usage(self, project_id, user_id=None):
    cmd = 'resource usage show %s' % project_id
    if user_id:
        cmd += ' --user-id %s' % user_id
    return self.openstack(cmd, use_json=True)