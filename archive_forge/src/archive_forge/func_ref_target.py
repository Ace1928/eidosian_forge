import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union
from urllib.parse import urlparse
from wandb.errors.term import termwarn
from wandb.sdk.lib import filesystem
from wandb.sdk.lib.hashutil import (
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath, URIStr
def ref_target(self) -> Union[FilePathStr, URIStr]:
    """Get the reference URL that is targeted by this artifact entry.

        Returns:
            (str): The reference URL of this artifact entry.

        Raises:
            ValueError: If this artifact entry was not a reference.
        """
    if self.ref is None:
        raise ValueError('Only reference entries support ref_target().')
    if self._parent_artifact is None:
        return self.ref
    return self._parent_artifact.manifest.storage_policy.load_reference(self._parent_artifact.manifest.entries[self.path], local=False)