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
def async_get_indexes(self, config, extra_hook=None, _app=None):
    """Asynchronous get indexes operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      extra_hook: Optional function to be called once the RPC has completed.

    Returns:
      A MultiRpc object.
    """
    req = datastore_pb.GetIndicesRequest()
    req.set_app_id(datastore_types.ResolveAppId(_app))
    resp = datastore_pb.CompositeIndices()
    return self._make_rpc_call(config, 'GetIndices', req, resp, get_result_hook=self.__get_indexes_hook, user_data=extra_hook, service_name=_DATASTORE_V3)