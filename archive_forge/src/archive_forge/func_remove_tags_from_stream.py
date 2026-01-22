import base64
import boto
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.kinesis import exceptions
from boto.compat import json
from boto.compat import six
def remove_tags_from_stream(self, stream_name, tag_keys):
    """
        Deletes tags from the specified Amazon Kinesis stream.

        If you specify a tag that does not exist, it is ignored.

        :type stream_name: string
        :param stream_name: The name of the stream.

        :type tag_keys: list
        :param tag_keys: A list of tag keys. Each corresponding tag is removed
            from the stream.

        """
    params = {'StreamName': stream_name, 'TagKeys': tag_keys}
    return self.make_request(action='RemoveTagsFromStream', body=json.dumps(params))