import logging
from typing import Optional, Tuple
import google.auth  # type: ignore
import google.cloud.artifactregistry  # type: ignore
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import (
from wandb.util import get_module
from .abstract import AbstractRegistry
Check if the image exists.

        Arguments:
            image_uri: The image URI.

        Returns:
            True if the image exists, False otherwise.
        