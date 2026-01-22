import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def set_topic_attributes(self, topic, attr_name, attr_value):
    """
        Get attributes of a Topic

        :type topic: string
        :param topic: The ARN of the topic.

        :type attr_name: string
        :param attr_name: The name of the attribute you want to set.
                          Only a subset of the topic's attributes are mutable.
                          Valid values: Policy | DisplayName

        :type attr_value: string
        :param attr_value: The new value for the attribute.

        """
    params = {'TopicArn': topic, 'AttributeName': attr_name, 'AttributeValue': attr_value}
    return self._make_request('SetTopicAttributes', params)