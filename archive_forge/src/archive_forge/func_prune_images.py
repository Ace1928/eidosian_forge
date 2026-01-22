import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
@utils.minimum_version('1.25')
def prune_images(self, filters=None):
    """
        Delete unused images

        Args:
            filters (dict): Filters to process on the prune list.
                Available filters:
                - dangling (bool):  When set to true (or 1), prune only
                unused and untagged images.

        Returns:
            (dict): A dict containing a list of deleted image IDs and
                the amount of disk space reclaimed in bytes.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    url = self._url('/images/prune')
    params = {}
    if filters is not None:
        params['filters'] = utils.convert_filters(filters)
    return self._result(self._post(url, params=params), True)