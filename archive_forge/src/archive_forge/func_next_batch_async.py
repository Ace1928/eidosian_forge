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
def next_batch_async(self, fetch_options=None):
    """Asynchronously get the next batch or None if there are no more batches.

    Args:
      fetch_options: Optional fetch options to use when fetching the next batch.
        Merged with both the fetch options on the original call and the
        connection.

    Returns:
      An async object that can be used to get the next Batch or None if either
      the next batch has already been fetched or there are no more results.
    """
    if not self.__datastore_cursor:
        return None
    fetch_options, next_batch = self._make_next_batch(fetch_options)
    if fetch_options is not None and (not FetchOptions.is_configuration(fetch_options)):
        raise datastore_errors.BadArgumentError('Invalid fetch options.')
    config = self._batch_shared.query_options.merge(fetch_options)
    conn = next_batch._batch_shared.conn
    requested_offset = 0
    if fetch_options is not None and fetch_options.offset is not None:
        requested_offset = fetch_options.offset
    if conn._api_version == datastore_rpc._CLOUD_DATASTORE_V1:
        if self._batch_shared.expected_offset != requested_offset:
            raise datastore_errors.BadArgumentError('Cannot request the next batch with a different offset than  expected. Expected: %s, Got: %s.' % (self._batch_shared.expected_offset, requested_offset))
        limit = self._batch_shared.remaining_limit
        next_options = QueryOptions(offset=self._batch_shared.expected_offset, limit=limit, start_cursor=self.__datastore_cursor)
        config = config.merge(next_options)
        result = next_batch._make_query_rpc_call(config, self._batch_shared.query._to_pb_v1(conn, config))
    else:
        result = next_batch._make_next_rpc_call(config, self._to_pb(fetch_options))
    self.__datastore_cursor = None
    return result