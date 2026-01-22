from typing import TYPE_CHECKING, Dict, List, Mapping, Optional
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.lib.hashutil import HexMD5
def remove_entry(self, entry: 'ArtifactManifestEntry') -> None:
    if entry.path not in self.entries:
        raise FileNotFoundError(f"Cannot remove missing entry: '{entry.path}'")
    del self.entries[entry.path]