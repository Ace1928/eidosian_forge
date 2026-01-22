from __future__ import annotations
from dataclasses import dataclass, field
import huggingface_hub
import semantic_version
import semantic_version as semver
def get_theme_assets(space_info: huggingface_hub.hf_api.SpaceInfo) -> list[ThemeAsset]:
    return [ThemeAsset(filename.rfilename) for filename in space_info.siblings if filename.rfilename.startswith('themes/')]