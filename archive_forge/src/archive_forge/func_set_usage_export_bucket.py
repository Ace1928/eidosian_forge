import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def set_usage_export_bucket(self, bucket, prefix=None):
    """
        Used to retain Compute Engine resource usage, storing the CSV data in
        a Google Cloud Storage bucket. See the
        `docs <https://cloud.google.com/compute/docs/usage-export>`_ for more
        information. Please ensure you have followed the necessary setup steps
        prior to enabling this feature (e.g. bucket exists, ACLs are in place,
        etc.)

        :param  bucket: Name of the Google Cloud Storage bucket. Specify the
                        name in either 'gs://<bucket_name>' or the full URL
                        'https://storage.googleapis.com/<bucket_name>'.
        :type   bucket: ``str``

        :param  prefix: Optional prefix string for all reports.
        :type   prefix: ``str`` or ``None``

        :return: True if successful
        :rtype:  ``bool``
        """
    return self.driver.ex_set_usage_export_bucket(bucket=bucket, prefix=prefix)