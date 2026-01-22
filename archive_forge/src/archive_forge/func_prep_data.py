from __future__ import annotations
from contextlib import nullcontext
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Collection, Literal, Sequence, cast
import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.color_util import (
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import Error, StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
from streamlit.runtime.metrics_util import gather_metrics
def prep_data(df: pd.DataFrame, x_column: str | None, y_column_list: list[str], color_column: str | None, size_column: str | None) -> tuple[pd.DataFrame, str | None, str | None, str | None, str | None]:
    """Prepares the data for charting. This is also used in add_rows.

    Returns the prepared dataframe and the new names of the x column (taking the index reset into
    consideration) and y, color, and size columns.
    """
    x_column = _maybe_reset_index_in_place(df, x_column, y_column_list)
    selected_data = _drop_unused_columns(df, x_column, color_column, size_column, *y_column_list)
    _maybe_convert_color_column_in_place(selected_data, color_column)
    x_column, y_column_list, color_column, size_column = _convert_col_names_to_str_in_place(selected_data, x_column, y_column_list, color_column, size_column)
    melted_data, y_column, color_column = _maybe_melt(selected_data, x_column, y_column_list, color_column, size_column)
    return (melted_data, x_column, y_column, color_column, size_column)