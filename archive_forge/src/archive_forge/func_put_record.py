import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def put_record(self, stream_name, data, partition_key, explicit_hash_key=None, sequence_number_for_ordering=None, exclusive_minimum_sequence_number=None, b64_encode=True):
    """
        This operation puts a data record into an Amazon Kinesis
        stream from a producer. This operation must be called to send
        data from the producer into the Amazon Kinesis stream for
        real-time ingestion and subsequent processing. The `PutRecord`
        operation requires the name of the stream that captures,
        stores, and transports the data; a partition key; and the data
        blob itself. The data blob could be a segment from a log file,
        geographic/location data, website clickstream data, or any
        other data type.

        The partition key is used to distribute data across shards.
        Amazon Kinesis segregates the data records that belong to a
        data stream into multiple shards, using the partition key
        associated with each data record to determine which shard a
        given data record belongs to.

        Partition keys are Unicode strings, with a maximum length
        limit of 256 bytes. An MD5 hash function is used to map
        partition keys to 128-bit integer values and to map associated
        data records to shards using the hash key ranges of the
        shards. You can override hashing the partition key to
        determine the shard by explicitly specifying a hash value
        using the `ExplicitHashKey` parameter. For more information,
        see the `Amazon Kinesis Developer Guide`_.

        `PutRecord` returns the shard ID of where the data record was
        placed and the sequence number that was assigned to the data
        record.

        Sequence numbers generally increase over time. To guarantee
        strictly increasing ordering, use the
        `SequenceNumberForOrdering` parameter. For more information,
        see the `Amazon Kinesis Developer Guide`_.

        If a `PutRecord` request cannot be processed because of
        insufficient provisioned throughput on the shard involved in
        the request, `PutRecord` throws
        `ProvisionedThroughputExceededException`.

        Data records are accessible for only 24 hours from the time
        that they are added to an Amazon Kinesis stream.

        :type stream_name: string
        :param stream_name: The name of the stream to put the data record into.

        :type data: blob
        :param data: The data blob to put into the record, which is
            Base64-encoded when the blob is serialized.
            The maximum size of the data blob (the payload after
            Base64-decoding) is 50 kilobytes (KB)
            Set `b64_encode` to disable automatic Base64 encoding.

        :type partition_key: string
        :param partition_key: Determines which shard in the stream the data
            record is assigned to. Partition keys are Unicode strings with a
            maximum length limit of 256 bytes. Amazon Kinesis uses the
            partition key as input to a hash function that maps the partition
            key and associated data to a specific shard. Specifically, an MD5
            hash function is used to map partition keys to 128-bit integer
            values and to map associated data records to shards. As a result of
            this hashing mechanism, all data records with the same partition
            key will map to the same shard within the stream.

        :type explicit_hash_key: string
        :param explicit_hash_key: The hash value used to explicitly determine
            the shard the data record is assigned to by overriding the
            partition key hash.

        :type sequence_number_for_ordering: string
        :param sequence_number_for_ordering: Guarantees strictly increasing
            sequence numbers, for puts from the same client and to the same
            partition key. Usage: set the `SequenceNumberForOrdering` of record
            n to the sequence number of record n-1 (as returned in the
            PutRecordResult when putting record n-1 ). If this parameter is not
            set, records will be coarsely ordered based on arrival time.

        :type b64_encode: boolean
        :param b64_encode: Whether to Base64 encode `data`. Can be set to
            ``False`` if `data` is already encoded to prevent double encoding.

        """
    params = {'StreamName': stream_name, 'Data': data, 'PartitionKey': partition_key}
    if explicit_hash_key is not None:
        params['ExplicitHashKey'] = explicit_hash_key
    if sequence_number_for_ordering is not None:
        params['SequenceNumberForOrdering'] = sequence_number_for_ordering
    if b64_encode:
        if not isinstance(params['Data'], six.binary_type):
            params['Data'] = params['Data'].encode('utf-8')
        params['Data'] = base64.b64encode(params['Data']).decode('utf-8')
    return self.make_request(action='PutRecord', body=json.dumps(params))