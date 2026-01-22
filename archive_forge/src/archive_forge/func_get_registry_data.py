import itertools
import re
import warnings
from ..api import APIClient
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..errors import BuildError, ImageLoadError, InvalidArgument
from ..utils import parse_repository_tag
from ..utils.json_stream import json_stream
from .resource import Collection, Model
def get_registry_data(self, name, auth_config=None):
    """
        Gets the registry data for an image.

        Args:
            name (str): The name of the image.
            auth_config (dict): Override the credentials that are found in the
                config for this request.  ``auth_config`` should contain the
                ``username`` and ``password`` keys to be valid.

        Returns:
            (:py:class:`RegistryData`): The data object.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    return RegistryData(image_name=name, attrs=self.client.api.inspect_distribution(name, auth_config), client=self.client, collection=self)