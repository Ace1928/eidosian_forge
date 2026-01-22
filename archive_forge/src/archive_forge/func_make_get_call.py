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
def make_get_call(base_req, pbs, extra_hook=None):
    req = copy.deepcopy(base_req)
    if self._api_version == _CLOUD_DATASTORE_V1:
        method = 'Lookup'
        req.keys.extend(pbs)
        resp = googledatastore.LookupResponse()
    else:
        method = 'Get'
        req.key_list().extend(pbs)
        resp = datastore_pb.GetResponse()
    user_data = (config, pbs, extra_hook)
    return self._make_rpc_call(config, method, req, resp, get_result_hook=self.__get_hook, user_data=user_data, service_name=self._api_version)