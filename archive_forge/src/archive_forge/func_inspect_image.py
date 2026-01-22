import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
@utils.check_resource('image')
def inspect_image(self, image):
    """
        Get detailed information about an image. Similar to the ``docker
        inspect`` command, but only for images.

        Args:
            image (str): The image to inspect

        Returns:
            (dict): Similar to the output of ``docker inspect``, but as a
        single dict

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    return self._result(self._get(self._url('/images/{0}/json', image)), True)