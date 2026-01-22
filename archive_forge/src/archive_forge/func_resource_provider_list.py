import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def resource_provider_list(self, uuid=None, name=None, aggregate_uuids=None, resources=None, in_tree=None, required=None, forbidden=None, member_of=None, may_print_to_stderr=False):
    to_exec = 'resource provider list'
    if uuid:
        to_exec += ' --uuid ' + uuid
    if name:
        to_exec += ' --name ' + name
    if aggregate_uuids:
        to_exec += ' ' + ' '.join(('--aggregate-uuid %s' % a for a in aggregate_uuids))
    if resources:
        to_exec += ' ' + ' '.join(('--resource %s' % r for r in resources))
    if in_tree:
        to_exec += ' --in-tree ' + in_tree
    if required:
        to_exec += ' ' + ' '.join(('--required %s' % t for t in required))
    if forbidden:
        to_exec += ' ' + ' '.join(('--forbidden %s' % f for f in forbidden))
    if member_of:
        to_exec += ' ' + ' '.join(['--member-of %s' % m for m in member_of])
    return self.openstack(to_exec, use_json=True, may_print_to_stderr=may_print_to_stderr)