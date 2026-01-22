import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def migration_start(self, share_id, dest_host, writable, nondisruptive, preserve_metadata, preserve_snapshots, force_host_assisted_migration, new_share_network=None, new_share_type=None):
    cmd = 'migration-start %(share)s %(host)s --writable %(writable)s --nondisruptive %(nondisruptive)s --preserve-metadata %(preserve_metadata)s --preserve-snapshots %(preserve_snapshots)s' % {'share': share_id, 'host': dest_host, 'writable': writable, 'nondisruptive': nondisruptive, 'preserve_metadata': preserve_metadata, 'preserve_snapshots': preserve_snapshots}
    if force_host_assisted_migration:
        cmd += ' --force-host-assisted-migration %s' % force_host_assisted_migration
    if new_share_network:
        cmd += ' --new-share-network %s' % new_share_network
    if new_share_type:
        cmd += ' --new-share-type %s' % new_share_type
    return self.manila(cmd)