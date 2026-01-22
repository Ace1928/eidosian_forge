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
def wait_for_resource_deletion(self, res_type, res_id, interval=3, timeout=180, microversion=None, **kwargs):
    """Resource deletion waiter.

        :param res_type: text -- type of resource
        :param res_id: text -- ID of resource to use for deletion check
        :param interval: int -- interval between requests in seconds
        :param timeout: int -- total time in seconds to wait for deletion
        :param args: dict -- additional keyword arguments for deletion func
        """
    if res_type == SHARE_TYPE:
        func = self.is_share_type_deleted
    elif res_type == SHARE_NETWORK:
        func = self.is_share_network_deleted
    elif res_type == SHARE_NETWORK_SUBNET:
        func = self.is_share_network_subnet_deleted
    elif res_type == SHARE_SERVER:
        func = self.is_share_server_deleted
    elif res_type == SHARE:
        func = self.is_share_deleted
    elif res_type == SNAPSHOT:
        func = self.is_snapshot_deleted
    elif res_type == MESSAGE:
        func = self.is_message_deleted
    elif res_type == SHARE_REPLICA:
        func = self.is_share_replica_deleted
    elif res_type == TRANSFER:
        func = self.is_share_transfer_deleted
    else:
        raise exceptions.InvalidResource(message=res_type)
    end_loop_time = time.time() + timeout
    deleted = func(res_id, microversion=microversion, **kwargs)
    while not (deleted or time.time() > end_loop_time):
        time.sleep(interval)
        deleted = func(res_id, microversion=microversion, **kwargs)
    if not deleted:
        raise exceptions.ResourceReleaseFailed(res_type=res_type, res_id=res_id)