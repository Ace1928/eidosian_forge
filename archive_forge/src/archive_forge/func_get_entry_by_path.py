from typing import TYPE_CHECKING, Dict, List, Mapping, Optional
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.lib.hashutil import HexMD5
def get_entry_by_path(self, path: str) -> Optional['ArtifactManifestEntry']:
    return self.entries.get(path)