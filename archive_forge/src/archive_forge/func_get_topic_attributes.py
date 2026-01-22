import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def get_topic_attributes(self, topic):
    """
        Get attributes of a Topic

        :type topic: string
        :param topic: The ARN of the topic.

        """
    params = {'TopicArn': topic}
    return self._make_request('GetTopicAttributes', params)