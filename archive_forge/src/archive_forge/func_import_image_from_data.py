import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
def import_image_from_data(self, data, repository=None, tag=None, changes=None):
    """
        Like :py:meth:`~docker.api.image.ImageApiMixin.import_image`, but
        allows importing in-memory bytes data.

        Args:
            data (bytes collection): Bytes collection containing valid tar data
            repository (str): The repository to create
            tag (str): The tag to apply
        """
    u = self._url('/images/create')
    params = _import_image_params(repository, tag, src='-', changes=changes)
    headers = {'Content-Type': 'application/tar'}
    return self._result(self._post(u, data=data, params=params, headers=headers, timeout=None))