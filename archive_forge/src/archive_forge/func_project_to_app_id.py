from __future__ import absolute_import
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_v4_pb
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
import six
def project_to_app_id(self, project_id):
    """Converts a string project id to a string app id."""
    return self._id_resolver.resolve_app_id(project_id)