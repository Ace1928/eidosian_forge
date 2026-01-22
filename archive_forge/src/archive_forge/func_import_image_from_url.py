import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
def import_image_from_url(self, url, repository=None, tag=None, changes=None):
    """
        Like :py:meth:`~docker.api.image.ImageApiMixin.import_image`, but only
        supports importing from a URL.

        Args:
            url (str): A URL pointing to a tar file.
            repository (str): The repository to create
            tag (str): The tag to apply
        """
    return self.import_image(src=url, repository=repository, tag=tag, changes=changes)