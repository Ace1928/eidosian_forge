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
def update_column_config(column_config_mapping: ColumnConfigMapping, column: str, column_config: ColumnConfig) -> None:
    """Updates the column config value for a single column within the mapping.

    Parameters
    ----------

    column_config_mapping : ColumnConfigMapping
        The column config mapping to update.

    column : str
        The column to update the config value for.

    column_config : ColumnConfig
        The column config to update.
    """
    if column not in column_config_mapping:
        column_config_mapping[column] = {}
    column_config_mapping[column].update(column_config)