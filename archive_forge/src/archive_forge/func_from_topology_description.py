from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
@classmethod
def from_topology_description(cls, topology_description: TopologyDescription) -> Selection:
    known_servers = topology_description.known_servers
    primary = None
    for sd in known_servers:
        if sd.server_type == SERVER_TYPE.RSPrimary:
            primary = sd
            break
    return Selection(topology_description, topology_description.known_servers, topology_description.common_wire_version, primary)