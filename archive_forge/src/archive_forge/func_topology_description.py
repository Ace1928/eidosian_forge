from __future__ import annotations
import contextlib
import os
import weakref
from collections import defaultdict
from typing import (
import bson
from bson.codec_options import DEFAULT_CODEC_OPTIONS, TypeRegistry
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import (
from pymongo.change_stream import ChangeStream, ClusterChangeStream
from pymongo.client_options import ClientOptions
from pymongo.client_session import _EmptyServerSession
from pymongo.command_cursor import CommandCursor
from pymongo.errors import (
from pymongo.lock import _HAS_REGISTER_AT_FORK, _create_lock, _release_locks
from pymongo.pool import ConnectionClosedReason
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_selectors import writable_server_selector
from pymongo.server_type import SERVER_TYPE
from pymongo.settings import TopologySettings
from pymongo.topology import Topology, _ErrorContext
from pymongo.topology_description import TOPOLOGY_TYPE, TopologyDescription
from pymongo.typings import (
from pymongo.uri_parser import (
from pymongo.write_concern import DEFAULT_WRITE_CONCERN, WriteConcern
@property
def topology_description(self) -> TopologyDescription:
    """The description of the connected MongoDB deployment.

        >>> client.topology_description
        <TopologyDescription id: 605a7b04e76489833a7c6113, topology_type: ReplicaSetWithPrimary, servers: [<ServerDescription ('localhost', 27017) server_type: RSPrimary, rtt: 0.0007973677999995488>, <ServerDescription ('localhost', 27018) server_type: RSSecondary, rtt: 0.0005540556000003249>, <ServerDescription ('localhost', 27019) server_type: RSSecondary, rtt: 0.0010367483999999649>]>
        >>> client.topology_description.topology_type_name
        'ReplicaSetWithPrimary'

        Note that the description is periodically updated in the background
        but the returned object itself is immutable. Access this property again
        to get a more recent
        :class:`~pymongo.topology_description.TopologyDescription`.

        :Returns:
          An instance of
          :class:`~pymongo.topology_description.TopologyDescription`.

        .. versionadded:: 4.0
        """
    return self._topology.description