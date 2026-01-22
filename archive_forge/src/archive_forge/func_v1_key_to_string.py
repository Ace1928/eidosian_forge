from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def v1_key_to_string(v1_key):
    """Generates a string representing a key's path.

  The output makes no effort to qualify special characters in strings.

  The key need not be valid, but if any of the key path elements have
  both a name and an ID the name is ignored.

  Args:
    v1_key: an googledatastore.Key

  Returns:
    a string representing the key's path
  """
    path_element_strings = []
    for path_element in v1_key.path:
        field = path_element.WhichOneof('id_type')
        if field == 'id':
            id_or_name = str(path_element.id)
        elif field == 'name':
            id_or_name = path_element.name
        else:
            id_or_name = ''
        path_element_strings.append('%s: %s' % (path_element.kind, id_or_name))
    return '[%s]' % ', '.join(path_element_strings)