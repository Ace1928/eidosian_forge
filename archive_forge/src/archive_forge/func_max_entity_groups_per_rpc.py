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
def max_entity_groups_per_rpc(value):
    """The maximum number of entity groups that can be represented in one rpc.

    For a non-transactional operation that involves more entity groups than the
    maximum, the operation will be performed by executing multiple, asynchronous
    rpcs to the datastore, each of which has no more entity groups represented
    than the maximum.  So, if a put() operation has 8 entity groups and the
    maximum is 3, we will send 3 rpcs, 2 with 3 entity groups and 1 with 2
    entity groups.  This is a performance optimization - in many cases
    multiple, small, concurrent rpcs will finish faster than a single large
    rpc.  The optimal value for this property will be application-specific, so
    experimentation is encouraged.
    """
    if not (isinstance(value, six_subset.integer_types) and value > 0):
        raise datastore_errors.BadArgumentError('max_entity_groups_per_rpc should be a positive integer')
    return value