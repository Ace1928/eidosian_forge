"""
Core version handling and configuration facilities.
"""
from .config import VersionConfig
from .version import (DEFAULT_VERSION, SimpleVersion, format_version,
                      parse_version)

__all__ = [
    'VersionConfig',
    'SimpleVersion',
    'parse_version',
    'format_version',
    'DEFAULT_VERSION',
]
