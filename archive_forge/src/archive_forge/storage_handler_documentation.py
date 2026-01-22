from typing import TYPE_CHECKING, Optional, Sequence, Union
from wandb.sdk.lib.paths import FilePathStr, URIStr
Store the file or directory at the given path to the specified artifact.

        Args:
            path: The path to store
            name: If specified, the logical name that should map to `path`
            checksum: Whether to compute the checksum of the file
            max_objects: The maximum number of objects to store

        Returns:
            A list of manifest entries to store within the artifact
        