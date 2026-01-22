import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
def import_image_from_file(self, filename, repository=None, tag=None, changes=None):
    """
        Like :py:meth:`~docker.api.image.ImageApiMixin.import_image`, but only
        supports importing from a tar file on disk.

        Args:
            filename (str): Full path to a tar file.
            repository (str): The repository to create
            tag (str): The tag to apply

        Raises:
            IOError: File does not exist.
        """
    return self.import_image(src=filename, repository=repository, tag=tag, changes=changes)