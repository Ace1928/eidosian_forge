from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
Yields bucket/notification tuples from command-line args.

  Given a list of strings that are bucket URLs ("gs://foo") or notification
  configuration URLs ("b/bucket/notificationConfigs/5"), yield tuples of
  bucket names and their associated notifications.

  Args:
    urls (list[str]): Bucket and notification configuration URLs to pull
      notification configurations from.
    accept_notification_configuration_urls (bool): Whether to raise an an error
      if a notification configuration URL is in `urls`.

  Yields:
    NotificationIteratorResult

  Raises:
    InvalidUrlError: Received notification configuration URL, but
      accept_notification_configuration_urls was False. Or received non-GCS
      bucket URL.
  