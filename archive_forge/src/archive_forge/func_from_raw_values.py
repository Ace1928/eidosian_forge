from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from textwrap import dedent
from typing import (
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.DateInput_pb2 import DateInput as DateInputProto
from streamlit.proto.TimeInput_pb2 import TimeInput as TimeInputProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.time_util import adjust_years
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
@classmethod
def from_raw_values(cls, value: DateValue | Literal['today'] | Literal['default_value_today'], min_value: SingleDateValue, max_value: SingleDateValue) -> _DateInputValues:
    parsed_value, is_range = _parse_date_value(value=value)
    return cls(value=parsed_value, is_range=is_range, min=_parse_min_date(min_value=min_value, parsed_dates=parsed_value), max=_parse_max_date(max_value=max_value, parsed_dates=parsed_value))