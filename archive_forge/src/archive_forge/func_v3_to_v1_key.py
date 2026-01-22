from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v3_to_v1_key(self, v3_ref, v1_key):
    """Converts a v3 Reference to a v1 Key.

    Args:
      v3_ref: an entity_pb.Reference
      v1_key: an googledatastore.Key to populate
    """
    v1_key.Clear()
    if not v3_ref.app():
        return
    project_id = self._id_resolver.resolve_project_id(v3_ref.app())
    v1_key.partition_id.project_id = project_id
    if v3_ref.name_space():
        v1_key.partition_id.namespace_id = v3_ref.name_space()
    for v3_element in v3_ref.path().element_list():
        v1_element = v1_key.path.add()
        v1_element.kind = v3_element.type()
        if v3_element.has_id():
            v1_element.id = v3_element.id()
        if v3_element.has_name():
            v1_element.name = v3_element.name()