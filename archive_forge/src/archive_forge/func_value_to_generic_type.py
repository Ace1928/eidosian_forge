from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from numbers import Integral, Real
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Final, Sequence, Tuple, TypeVar, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.js_number import JSNumber, JSNumberBoundsException
from streamlit.proto.Slider_pb2 import Slider as SliderProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
def value_to_generic_type(v):
    if isinstance(v, Integral):
        return SUPPORTED_TYPES[Integral]
    elif isinstance(v, Real):
        return SUPPORTED_TYPES[Real]
    else:
        return SUPPORTED_TYPES[type(v)]