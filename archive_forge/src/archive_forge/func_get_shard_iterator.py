import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def get_shard_iterator(self, stream_name, shard_id, shard_iterator_type, starting_sequence_number=None):
    """
        Gets a shard iterator. A shard iterator expires five minutes
        after it is returned to the requester.

        A shard iterator specifies the position in the shard from
        which to start reading data records sequentially. A shard
        iterator specifies this position using the sequence number of
        a data record in a shard. A sequence number is the identifier
        associated with every record ingested in the Amazon Kinesis
        stream. The sequence number is assigned when a record is put
        into the stream.

        You must specify the shard iterator type. For example, you can
        set the `ShardIteratorType` parameter to read exactly from the
        position denoted by a specific sequence number by using the
        `AT_SEQUENCE_NUMBER` shard iterator type, or right after the
        sequence number by using the `AFTER_SEQUENCE_NUMBER` shard
        iterator type, using sequence numbers returned by earlier
        calls to PutRecord, PutRecords, GetRecords, or DescribeStream.
        You can specify the shard iterator type `TRIM_HORIZON` in the
        request to cause `ShardIterator` to point to the last
        untrimmed record in the shard in the system, which is the
        oldest data record in the shard. Or you can point to just
        after the most recent record in the shard, by using the shard
        iterator type `LATEST`, so that you always read the most
        recent data in the shard.

        When you repeatedly read from an Amazon Kinesis stream use a
        GetShardIterator request to get the first shard iterator to to
        use in your first `GetRecords` request and then use the shard
        iterator returned by the `GetRecords` request in
        `NextShardIterator` for subsequent reads. A new shard iterator
        is returned by every `GetRecords` request in
        `NextShardIterator`, which you use in the `ShardIterator`
        parameter of the next `GetRecords` request.

        If a `GetShardIterator` request is made too often, you receive
        a `ProvisionedThroughputExceededException`. For more
        information about throughput limits, see GetRecords.

        If the shard is closed, the iterator can't return more data,
        and `GetShardIterator` returns `null` for its `ShardIterator`.
        A shard can be closed using SplitShard or MergeShards.

        `GetShardIterator` has a limit of 5 transactions per second
        per account per open shard.

        :type stream_name: string
        :param stream_name: The name of the stream.

        :type shard_id: string
        :param shard_id: The shard ID of the shard to get the iterator for.

        :type shard_iterator_type: string
        :param shard_iterator_type:
        Determines how the shard iterator is used to start reading data records
            from the shard.

        The following are the valid shard iterator types:


        + AT_SEQUENCE_NUMBER - Start reading exactly from the position denoted
              by a specific sequence number.
        + AFTER_SEQUENCE_NUMBER - Start reading right after the position
              denoted by a specific sequence number.
        + TRIM_HORIZON - Start reading at the last untrimmed record in the
              shard in the system, which is the oldest data record in the shard.
        + LATEST - Start reading just after the most recent record in the
              shard, so that you always read the most recent data in the shard.

        :type starting_sequence_number: string
        :param starting_sequence_number: The sequence number of the data record
            in the shard from which to start reading from.

        :returns: A dictionary containing:

            1) a `ShardIterator` with the value being the shard-iterator object
        """
    params = {'StreamName': stream_name, 'ShardId': shard_id, 'ShardIteratorType': shard_iterator_type}
    if starting_sequence_number is not None:
        params['StartingSequenceNumber'] = starting_sequence_number
    return self.make_request(action='GetShardIterator', body=json.dumps(params))