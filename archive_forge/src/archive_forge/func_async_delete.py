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
def async_delete(self, config, keys, extra_hook=None):
    """Transactional asynchronous Delete operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      keys: An iterable of user-level key objects.
      extra_hook: Optional function to be called once the RPC has completed.

    Returns:
      A MultiRpc object.
    """
    if self._api_version != _CLOUD_DATASTORE_V1:
        return super(TransactionalConnection, self).async_delete(config, keys, extra_hook)
    v1_keys = [self.__adapter.key_to_pb_v1(key) for key in keys]
    for key in v1_keys:
        hashable_key = datastore_types.ReferenceToKeyValue(key)
        self.__pending_v1_upserts.pop(hashable_key, None)
        self.__pending_v1_deletes[hashable_key] = key
    return self._make_rpc_call(config, 'Commit', None, googledatastore.CommitResponse(), get_result_hook=self.__v1_delete_hook, user_data=extra_hook, service_name=_NOOP_SERVICE)