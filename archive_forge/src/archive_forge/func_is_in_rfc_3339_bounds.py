from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def is_in_rfc_3339_bounds(microseconds):
    return RFC_3339_MIN_MICROSECONDS_INCLUSIVE <= microseconds <= RFC_3339_MAX_MICROSECONDS_INCLUSIVE