from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
def selection_hook(self, topology_description: TopologyDescription) -> None:
    common_wv = topology_description.common_wire_version
    if topology_description.has_readable_server(ReadPreference.PRIMARY_PREFERRED) and common_wv and (common_wv < 13):
        self.effective_pref = ReadPreference.PRIMARY
    else:
        self.effective_pref = self.pref