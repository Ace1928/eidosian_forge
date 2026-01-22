from __future__ import annotations
import json
from enum import Enum
from typing import TYPE_CHECKING, Dict, Final, Literal, Mapping, Union
from typing_extensions import TypeAlias
from streamlit.elements.lib.column_types import ColumnConfig, ColumnType
from streamlit.elements.lib.dicttools import remove_none_values
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.type_util import DataFormat, is_colum_type_arrow_incompatible
def determine_dataframe_schema(data_df: DataFrame, arrow_schema: pa.Schema) -> DataframeSchema:
    """Determine the schema of a dataframe.

    Parameters
    ----------
    data_df : pd.DataFrame
        The dataframe to determine the schema of.
    arrow_schema : pa.Schema
        The Arrow schema of the dataframe.

    Returns
    -------

    DataframeSchema
        A mapping that contains the detected data type for the index and columns.
        The key is the column name in the underlying dataframe or ``_index`` for index columns.
    """
    dataframe_schema: DataframeSchema = {}
    dataframe_schema[INDEX_IDENTIFIER] = _determine_data_kind(data_df.index)
    for i, column in enumerate(data_df.items()):
        column_name, column_data = column
        dataframe_schema[column_name] = _determine_data_kind(column_data, arrow_schema.field(i))
    return dataframe_schema