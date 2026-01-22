from typing import TYPE_CHECKING, Dict, List, Mapping, Optional
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.lib.hashutil import HexMD5
def get_entries_in_directory(self, directory: str) -> List['ArtifactManifestEntry']:
    return [self.entries[entry_key] for entry_key in self.entries if entry_key.startswith(directory + '/')]