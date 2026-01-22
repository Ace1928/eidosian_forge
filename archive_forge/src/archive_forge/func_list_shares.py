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
def list_shares(self, all_tenants=False, is_soft_deleted=False, filters=None, columns=None, is_public=False, microversion=None):
    """List shares.

        :param all_tenants: bool -- whether to list shares that belong
            only to current project or for all tenants.
        :param is_soft_deleted: bool -- whether to list shares that has
            been soft deleted to recycle bin.
        :param filters: dict -- filters for listing of shares.
            Example, input:
                {'project_id': 'foo'}
                {-'project_id': 'foo'}
                {--'project_id': 'foo'}
                {'project-id': 'foo'}
            will be transformed to filter parameter "--project-id=foo"
        :param columns: comma separated string of columns.
            Example, "--columns Name,Size"
        :param is_public: bool -- should list public shares or not.
            Default is False.
        :param microversion: str -- the request api version.
        """
    cmd = 'list '
    if all_tenants:
        cmd += '--all-tenants '
    if is_public:
        cmd += '--public '
    if is_soft_deleted:
        cmd += '--soft-deleted '
    if filters and isinstance(filters, dict):
        for k, v in filters.items():
            cmd += '%(k)s=%(v)s ' % {'k': self._stranslate_to_cli_optional_param(k), 'v': v}
    if columns is not None:
        cmd += '--columns ' + columns
    shares_raw = self.manila(cmd, microversion=microversion)
    shares = utils.listing(shares_raw)
    return shares