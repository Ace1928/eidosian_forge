from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from dateutil import tz
from googlecloudsdk.core.util import times
Use this function in a display transform to truncate anything smaller than minutes from ISO8601 timestamp.