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
def on_script_will_rerun(self, latest_widget_states: WidgetStatesProto) -> None:
    """Called by ScriptRunner before its script re-runs.

        Update widget data and call callbacks on widgets whose value changed
        between the previous and current script runs.
        """
    self._reset_triggers()
    self._compact_state()
    self.set_widgets_from_proto(latest_widget_states)
    self._call_callbacks()