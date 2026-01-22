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
@ConfigOption
def on_completion(value):
    """A callback that is invoked when any RPC completes.

    If specified, it will be called with a UserRPC object as argument when an
    RPC completes.

    NOTE: There is a subtle but important difference between
    UserRPC.callback and Configuration.on_completion: on_completion is
    called with the RPC object as its first argument, where callback is
    called without arguments.  (Because a Configuration's on_completion
    function can be used with many UserRPC objects, it would be awkward
    if it was called without passing the specific RPC.)
    """
    return value