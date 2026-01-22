from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.cloud.pubsublite import cloudpubsub
from google.cloud.pubsublite import types
from google.cloud.pubsublite.cloudpubsub import message_transforms
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.command_lib.pubsub import lite_util
from googlecloudsdk.core import gapic_util
from googlecloudsdk.core.util import http_encoding
Publishes a message to the specified Pub/Sub Lite topic.

    Args:
      topic_resource: The pubsub.lite_topic resource to publish to.
      message: The string message to publish.
      ordering_key: The key for ordering delivery to subscribers.
      attributes: A dict of attributes to attach to the message.
      event_time: A user-specified event timestamp.

    Raises:
      EmptyMessageException: if the message is empty.
      PublishOperationException: if the publish operation is not successful.

    Returns:
      The messageId of the published message, containing the Partition and
      Offset.
    