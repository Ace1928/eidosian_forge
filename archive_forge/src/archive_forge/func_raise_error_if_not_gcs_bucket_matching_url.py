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
def raise_error_if_not_gcs_bucket_matching_url(url):
    """Raises error if URL is not supported for notifications."""
    if not (url.scheme is storage_url.ProviderPrefix.GCS and (url.is_bucket() or url.is_provider())):
        raise errors.InvalidUrlError('Notification configurations available on only Google Cloud Storage buckets. Invalid URL: ' + url.url_string)