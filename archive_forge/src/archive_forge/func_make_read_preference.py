from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
def make_read_preference(mode: int, tag_sets: Optional[_TagSets], max_staleness: int=-1) -> _ServerMode:
    if mode == _PRIMARY:
        if tag_sets not in (None, [{}]):
            raise ConfigurationError('Read preference primary cannot be combined with tags')
        if max_staleness != -1:
            raise ConfigurationError('Read preference primary cannot be combined with maxStalenessSeconds')
        return Primary()
    return _ALL_READ_PREFERENCES[mode](tag_sets, max_staleness)