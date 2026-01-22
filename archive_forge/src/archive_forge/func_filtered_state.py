from __future__ import annotations
import json
import pickle
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import (
from typing_extensions import TypeAlias
import streamlit as st
from streamlit import config, util
from streamlit.errors import StreamlitAPIException, UnserializableSessionStateError
from streamlit.proto.WidgetStates_pb2 import WidgetState as WidgetStateProto
from streamlit.proto.WidgetStates_pb2 import WidgetStates as WidgetStatesProto
from streamlit.runtime.state.common import (
from streamlit.runtime.state.query_params import QueryParams
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.type_util import ValueFieldName, is_array_value_field_name
@property
def filtered_state(self) -> dict[str, Any]:
    """The combined session and widget state, excluding keyless widgets."""
    wid_key_map = self._reverse_key_wid_map
    state: dict[str, Any] = {}
    for k in self._keys():
        if not is_widget_id(k) and (not _is_internal_key(k)):
            state[k] = self[k]
        elif is_keyed_widget_id(k):
            try:
                key = wid_key_map[k]
                state[key] = self[k]
            except KeyError:
                pass
    return state