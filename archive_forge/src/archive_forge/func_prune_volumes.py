from .. import errors
from .. import utils
@utils.minimum_version('1.25')
def prune_volumes(self, filters=None):
    """
        Delete unused volumes

        Args:
            filters (dict): Filters to process on the prune list.

        Returns:
            (dict): A dict containing a list of deleted volume names and
                the amount of disk space reclaimed in bytes.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    params = {}
    if filters:
        params['filters'] = utils.convert_filters(filters)
    url = self._url('/volumes/prune')
    return self._result(self._post(url, params=params), True)