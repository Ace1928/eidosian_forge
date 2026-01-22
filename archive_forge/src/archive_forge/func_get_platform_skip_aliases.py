from __future__ import annotations
import abc
import typing as t
from ...config import (
from ...util import (
from ...target import (
from ...host_configs import (
from ...host_profiles import (
def get_platform_skip_aliases(platform: str, version: str, arch: t.Optional[str]) -> dict[str, str]:
    """Return a dictionary of skip aliases and the reason why they apply."""
    skips = {f'skip/{platform}': platform, f'skip/{platform}/{version}': f'{platform} {version}', f'skip/{platform}{version}': f'{platform} {version}'}
    if arch:
        skips.update({f'skip/{arch}': arch, f'skip/{arch}/{platform}': f'{platform} on {arch}', f'skip/{arch}/{platform}/{version}': f'{platform} {version} on {arch}'})
    skips = {alias: f'which are not supported by {description}' for alias, description in skips.items()}
    return skips