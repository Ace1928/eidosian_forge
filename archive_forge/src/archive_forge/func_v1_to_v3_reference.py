from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v1_to_v3_reference(self, v1_key, v3_ref):
    """Converts a v1 Key to a v3 Reference.

    Args:
      v1_key: an googledatastore.Key
      v3_ref: an entity_pb.Reference to populate
    """
    v3_ref.Clear()
    if v1_key.HasField('partition_id'):
        project_id = v1_key.partition_id.project_id
        if project_id:
            app_id = self._id_resolver.resolve_app_id(project_id)
            v3_ref.set_app(app_id)
        if v1_key.partition_id.namespace_id:
            v3_ref.set_name_space(v1_key.partition_id.namespace_id)
    for v1_element in v1_key.path:
        v3_element = v3_ref.mutable_path().add_element()
        v3_element.set_type(v1_element.kind.encode('utf-8'))
        id_type = v1_element.WhichOneof('id_type')
        if id_type == 'id':
            v3_element.set_id(v1_element.id)
        elif id_type == 'name':
            v3_element.set_name(v1_element.name.encode('utf-8'))