import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def split_shard(self, stream_name, shard_to_split, new_starting_hash_key):
    """
        Splits a shard into two new shards in the stream, to increase
        the stream's capacity to ingest and transport data.
        `SplitShard` is called when there is a need to increase the
        overall capacity of stream because of an expected increase in
        the volume of data records being ingested.

        You can also use `SplitShard` when a shard appears to be
        approaching its maximum utilization, for example, when the set
        of producers sending data into the specific shard are suddenly
        sending more than previously anticipated. You can also call
        `SplitShard` to increase stream capacity, so that more Amazon
        Kinesis applications can simultaneously read data from the
        stream for real-time processing.

        You must specify the shard to be split and the new hash key,
        which is the position in the shard where the shard gets split
        in two. In many cases, the new hash key might simply be the
        average of the beginning and ending hash key, but it can be
        any hash key value in the range being mapped into the shard.
        For more information about splitting shards, see `Split a
        Shard`_ in the Amazon Kinesis Developer Guide .

        You can use DescribeStream to determine the shard ID and hash
        key values for the `ShardToSplit` and `NewStartingHashKey`
        parameters that are specified in the `SplitShard` request.

        `SplitShard` is an asynchronous operation. Upon receiving a
        `SplitShard` request, Amazon Kinesis immediately returns a
        response and sets the stream status to `UPDATING`. After the
        operation is completed, Amazon Kinesis sets the stream status
        to `ACTIVE`. Read and write operations continue to work while
        the stream is in the `UPDATING` state.

        You can use `DescribeStream` to check the status of the
        stream, which is returned in `StreamStatus`. If the stream is
        in the `ACTIVE` state, you can call `SplitShard`. If a stream
        is in `CREATING` or `UPDATING` or `DELETING` states,
        `DescribeStream` returns a `ResourceInUseException`.

        If the specified stream does not exist, `DescribeStream`
        returns a `ResourceNotFoundException`. If you try to create
        more shards than are authorized for your account, you receive
        a `LimitExceededException`.

        The default limit for an AWS account is 10 shards per stream.
        If you need to create a stream with more than 10 shards,
        `contact AWS Support`_ to increase the limit on your account.

        If you try to operate on too many streams in parallel using
        CreateStream, DeleteStream, MergeShards or SplitShard, you
        receive a `LimitExceededException`.

        `SplitShard` has limit of 5 transactions per second per
        account.

        :type stream_name: string
        :param stream_name: The name of the stream for the shard split.

        :type shard_to_split: string
        :param shard_to_split: The shard ID of the shard to split.

        :type new_starting_hash_key: string
        :param new_starting_hash_key: A hash key value for the starting hash
            key of one of the child shards created by the split. The hash key
            range for a given shard constitutes a set of ordered contiguous
            positive integers. The value for `NewStartingHashKey` must be in
            the range of hash keys being mapped into the shard. The
            `NewStartingHashKey` hash key value and all higher hash key values
            in hash key range are distributed to one of the child shards. All
            the lower hash key values in the range are distributed to the other
            child shard.

        """
    params = {'StreamName': stream_name, 'ShardToSplit': shard_to_split, 'NewStartingHashKey': new_starting_hash_key}
    return self.make_request(action='SplitShard', body=json.dumps(params))