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
def update_share_type(self, share_type_id, name=None, is_public=None, microversion=None, description=None):
    """Update share type.

        :param share_type_id: text -- id of share type.
        :param name: text -- new name of share type, if not set then
            it will not be updated.
        :param description: text -- new description of share type.
            if not set then it will not be updated.
        :param is_public: bool/str -- boolean or its string alias.
            new visibility of the share type.If set to True, share
            type will be available to all tenants in the cloud.
        """
    cmd = 'type-update %(share_type_id)s ' % {'share_type_id': share_type_id}
    if is_public is not None:
        if not isinstance(is_public, str):
            is_public = str(is_public)
        cmd += ' --is_public ' + is_public
    if description:
        cmd += ' --description ' + description
    elif description == '':
        cmd += ' --description "" '
    if name:
        cmd += ' --name ' + name
    share_type_raw = self.manila(cmd, microversion=microversion)
    share_type = utils.details(share_type_raw)
    return share_type