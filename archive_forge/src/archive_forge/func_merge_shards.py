import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def merge_shards(self, stream_name, shard_to_merge, adjacent_shard_to_merge):
    """
        Merges two adjacent shards in a stream and combines them into
        a single shard to reduce the stream's capacity to ingest and
        transport data. Two shards are considered adjacent if the
        union of the hash key ranges for the two shards form a
        contiguous set with no gaps. For example, if you have two
        shards, one with a hash key range of 276...381 and the other
        with a hash key range of 382...454, then you could merge these
        two shards into a single shard that would have a hash key
        range of 276...454. After the merge, the single child shard
        receives data for all hash key values covered by the two
        parent shards.

        `MergeShards` is called when there is a need to reduce the
        overall capacity of a stream because of excess capacity that
        is not being used. You must specify the shard to be merged and
        the adjacent shard for a stream. For more information about
        merging shards, see `Merge Two Shards`_ in the Amazon Kinesis
        Developer Guide .

        If the stream is in the `ACTIVE` state, you can call
        `MergeShards`. If a stream is in the `CREATING`, `UPDATING`,
        or `DELETING` state, `MergeShards` returns a
        `ResourceInUseException`. If the specified stream does not
        exist, `MergeShards` returns a `ResourceNotFoundException`.

        You can use DescribeStream to check the state of the stream,
        which is returned in `StreamStatus`.

        `MergeShards` is an asynchronous operation. Upon receiving a
        `MergeShards` request, Amazon Kinesis immediately returns a
        response and sets the `StreamStatus` to `UPDATING`. After the
        operation is completed, Amazon Kinesis sets the `StreamStatus`
        to `ACTIVE`. Read and write operations continue to work while
        the stream is in the `UPDATING` state.

        You use DescribeStream to determine the shard IDs that are
        specified in the `MergeShards` request.

        If you try to operate on too many streams in parallel using
        CreateStream, DeleteStream, `MergeShards` or SplitShard, you
        will receive a `LimitExceededException`.

        `MergeShards` has limit of 5 transactions per second per
        account.

        :type stream_name: string
        :param stream_name: The name of the stream for the merge.

        :type shard_to_merge: string
        :param shard_to_merge: The shard ID of the shard to combine with the
            adjacent shard for the merge.

        :type adjacent_shard_to_merge: string
        :param adjacent_shard_to_merge: The shard ID of the adjacent shard for
            the merge.

        """
    params = {'StreamName': stream_name, 'ShardToMerge': shard_to_merge, 'AdjacentShardToMerge': adjacent_shard_to_merge}
    return self.make_request(action='MergeShards', body=json.dumps(params))