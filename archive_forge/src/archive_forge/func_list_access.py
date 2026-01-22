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
@not_found_wrapper
def list_access(self, entity_id, columns=None, microversion=None, is_snapshot=False, metadata=None):
    """Returns list of access rules for a share.

        :param entity_id: str -- Name or ID of a share or snapshot.
        :param columns: comma separated string of columns.
            Example, "--columns access_type,access_to"
        :param is_snapshot: Boolean value to determine if should list
            access of a share or snapshot.
        """
    if is_snapshot:
        cmd = 'snapshot-access-list %s ' % entity_id
    else:
        cmd = 'access-list %s ' % entity_id
    if columns is not None:
        cmd += ' --columns ' + columns
    if metadata:
        metadata_cli = ''
        for k, v in metadata.items():
            metadata_cli += '%(k)s=%(v)s ' % {'k': k, 'v': v}
        if metadata_cli:
            cmd += ' --metadata %s ' % metadata_cli
    access_list_raw = self.manila(cmd, microversion=microversion)
    return output_parser.listing(access_list_raw)