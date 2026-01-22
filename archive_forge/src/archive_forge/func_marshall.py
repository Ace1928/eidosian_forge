from __future__ import annotations
from typing import TYPE_CHECKING, Any
from streamlit import type_util
from streamlit.elements.lib import pandas_styler_utils
from streamlit.proto.Components_pb2 import ArrowTable as ArrowTableProto
def marshall(proto: ArrowTableProto, data: Any, default_uuid: str | None=None) -> None:
    """Marshall data into an ArrowTable proto.

    Parameters
    ----------
    proto : proto.ArrowTable
        Output. The protobuf for a Streamlit ArrowTable proto.

    data : pandas.DataFrame, pandas.Styler, numpy.ndarray, Iterable, dict, or None
        Something that is or can be converted to a dataframe.

    """
    if type_util.is_pandas_styler(data):
        pandas_styler_utils.marshall_styler(proto, data, default_uuid)
    df = type_util.convert_anything_to_df(data)
    _marshall_index(proto, df.index)
    _marshall_columns(proto, df.columns)
    _marshall_data(proto, df)