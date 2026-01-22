from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import exceptions
from googlecloudsdk.core.cache import persistent_cache_base
import six
Returns the list of unrestricted table names matching name.

    Args:
      name: The table name pattern. None for all unrestricted tables. May
        contain the * and ? pattern match characters.

    Returns:
      The list of unrestricted table names matching name.
    