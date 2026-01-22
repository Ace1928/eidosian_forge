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
def list_share_servers(self, filters=None, columns=None, microversion=None):
    """List share servers.

        :param filters: dict -- filters for listing of share servers.
            Example, input:
                {'project_id': 'foo'}
                {'-project_id': 'foo'}
                {'--project_id': 'foo'}
                {'project-id': 'foo'}
            will be transformed to filter parameter "--project-id=foo"
         :param columns: comma separated string of columns.
            Example, "--columns id"
        """
    cmd = 'share-server-list '
    if columns is not None:
        cmd += ' --columns ' + columns
    if filters and isinstance(filters, dict):
        for k, v in filters.items():
            cmd += '%(k)s=%(v)s ' % {'k': self._stranslate_to_cli_optional_param(k), 'v': v}
    share_servers_raw = self.manila(cmd, microversion=microversion)
    share_servers = utils.listing(share_servers_raw)
    return share_servers