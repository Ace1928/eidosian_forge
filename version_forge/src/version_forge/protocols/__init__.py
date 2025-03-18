"""
Protocol definitions for version_forge typing.
"""
from .interfaces import (ConfigSource, IVersioned, VersionData, VersionDelta,
                         VersionDict, VersionLike, VersionProtocol,
                         VersionStatus, VersionUpdateResult)

__all__ = [
    'ConfigSource',
    'IVersioned',
    'VersionData',
    'VersionDelta',
    'VersionDict',
    'VersionLike',
    'VersionProtocol',
    'VersionStatus',
    'VersionUpdateResult',
]
