from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
@datastore_rpc._positional(1)
def inject_results(query, updated_entities=None, deleted_keys=None):
    """Creates a query object that will inject changes into results.

  Args:
    query: The datastore_query.Query to augment
    updated_entities: A list of entity_pb.EntityProto's that have been updated
      and should take priority over any values returned by query.
    deleted_keys: A list of entity_pb.Reference's for entities that have been
      deleted and should be removed from query results.

  Returns:
    A datastore_query.AugmentedQuery if in memory filtering is required,
  query otherwise.
  """
    if not isinstance(query, Query):
        raise datastore_errors.BadArgumentError('query argument should be datastore_query.Query (%r)' % (query,))
    overridden_keys = set()
    if deleted_keys is not None:
        if not isinstance(deleted_keys, list):
            raise datastore_errors.BadArgumentError('deleted_keys argument must be a list (%r)' % (deleted_keys,))
        deleted_keys = list(filter(query._key_filter, deleted_keys))
        for key in deleted_keys:
            overridden_keys.add(datastore_types.ReferenceToKeyValue(key))
    if updated_entities is not None:
        if not isinstance(updated_entities, list):
            raise datastore_errors.BadArgumentError('updated_entities argument must be a list (%r)' % (updated_entities,))
        updated_entities = list(filter(query._key_filter, updated_entities))
        for entity in updated_entities:
            overridden_keys.add(datastore_types.ReferenceToKeyValue(entity.key()))
        updated_entities = apply_query(query, updated_entities)
    else:
        updated_entities = []
    if not overridden_keys:
        return query
    return _AugmentedQuery(query, in_memory_filter=_IgnoreFilter(overridden_keys), in_memory_results=updated_entities, max_filtered_count=len(overridden_keys))