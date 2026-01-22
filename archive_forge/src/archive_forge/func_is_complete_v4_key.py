from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def is_complete_v4_key(v4_key):
    """Returns True if a key specifies an ID or name, False otherwise.

  Args:
    v4_key: an entity_v4_pb.Key

  Returns:
    True if the key specifies an ID or name, False otherwise.
  """
    assert len(v4_key.path_element_list()) >= 1
    last_element = v4_key.path_element(len(v4_key.path_element_list()) - 1)
    return last_element.has_id() or last_element.has_name()