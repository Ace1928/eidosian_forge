from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_to_v1_keys(self, v3_refs):
    """Converts a list of v3 References to a list of v1 Keys.

    Args:
      v3_refs: a list of entity_pb.Reference objects

    Returns:
      a list of googledatastore.Key objects
    """
    v1_keys = []
    for v3_ref in v3_refs:
        v1_key = googledatastore.Key()
        self.v3_to_v1_key(v3_ref, v1_key)
        v1_keys.append(v1_key)
    return v1_keys