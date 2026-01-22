from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
def async_rollback(self, config):
    """Asynchronous Rollback operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.

     Returns:
      A MultiRpc object.
    """
    self.wait_for_all_pending_rpcs()
    if not (self._state == TransactionalConnection.OPEN or self._state == TransactionalConnection.FAILED):
        raise datastore_errors.BadRequestError('Cannot rollback transaction that is neither OPEN or FAILED state.')
    transaction = self.transaction
    if transaction is None:
        return None
    self._state = TransactionalConnection.CLOSED
    self.__transaction = None
    if self._api_version == _CLOUD_DATASTORE_V1:
        req = googledatastore.RollbackRequest()
        req.transaction = transaction
        resp = googledatastore.RollbackResponse()
    else:
        req = transaction
        resp = api_base_pb.VoidProto()
    return self._make_rpc_call(config, 'Rollback', req, resp, get_result_hook=self.__rollback_hook, service_name=self._api_version)