from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.api_lib.recommender import flag_utils
Mark a recommendation's state as FAILED.

    Args:
      name: str, the name of the recommendation being updated.
      state_metadata: A map of metadata for the state, provided by user or
        automations systems.
      etag: Fingerprint of the Recommendation. Provides optimistic locking when
        updating states.

    Returns:
      The result recommendations after being marked as accepted
    