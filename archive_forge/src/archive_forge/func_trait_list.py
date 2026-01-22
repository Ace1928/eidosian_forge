import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def trait_list(self, name=None, associated=False):
    cmd = 'trait list'
    if name:
        cmd += ' --name ' + name
    if associated:
        cmd += ' --associated'
    return self.openstack(cmd, use_json=True)