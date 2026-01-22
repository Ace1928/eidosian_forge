import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def delete_stream(self, stream_name):
    """
        Deletes a stream and all its shards and data. You must shut
        down any applications that are operating on the stream before
        you delete the stream. If an application attempts to operate
        on a deleted stream, it will receive the exception
        `ResourceNotFoundException`.

        If the stream is in the `ACTIVE` state, you can delete it.
        After a `DeleteStream` request, the specified stream is in the
        `DELETING` state until Amazon Kinesis completes the deletion.

        **Note:** Amazon Kinesis might continue to accept data read
        and write operations, such as PutRecord, PutRecords, and
        GetRecords, on a stream in the `DELETING` state until the
        stream deletion is complete.

        When you delete a stream, any shards in that stream are also
        deleted, and any tags are dissociated from the stream.

        You can use the DescribeStream operation to check the state of
        the stream, which is returned in `StreamStatus`.

        `DeleteStream` has a limit of 5 transactions per second per
        account.

        :type stream_name: string
        :param stream_name: The name of the stream to delete.

        """
    params = {'StreamName': stream_name}
    return self.make_request(action='DeleteStream', body=json.dumps(params))