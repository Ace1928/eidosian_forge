import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def list_streams(self, limit=None, exclusive_start_stream_name=None):
    """
        Lists your streams.

        The number of streams may be too large to return from a single
        call to `ListStreams`. You can limit the number of returned
        streams using the `Limit` parameter. If you do not specify a
        value for the `Limit` parameter, Amazon Kinesis uses the
        default limit, which is currently 10.

        You can detect if there are more streams available to list by
        using the `HasMoreStreams` flag from the returned output. If
        there are more streams available, you can request more streams
        by using the name of the last stream returned by the
        `ListStreams` request in the `ExclusiveStartStreamName`
        parameter in a subsequent request to `ListStreams`. The group
        of stream names returned by the subsequent request is then
        added to the list. You can continue this process until all the
        stream names have been collected in the list.

        `ListStreams` has a limit of 5 transactions per second per
        account.

        :type limit: integer
        :param limit: The maximum number of streams to list.

        :type exclusive_start_stream_name: string
        :param exclusive_start_stream_name: The name of the stream to start the
            list with.

        """
    params = {}
    if limit is not None:
        params['Limit'] = limit
    if exclusive_start_stream_name is not None:
        params['ExclusiveStartStreamName'] = exclusive_start_stream_name
    return self.make_request(action='ListStreams', body=json.dumps(params))