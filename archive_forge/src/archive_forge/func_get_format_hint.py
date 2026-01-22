from typing import Any, Optional
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame, ArrayDataFrame
from fugue.extensions.context import ExtensionContext
from fugue.extensions.transformer.constants import OUTPUT_TRANSFORMER_DUMMY_SCHEMA
def get_format_hint(self) -> Optional[str]:
    """Get the transformer's preferred data format, for example it can be
        ``pandas``, ``pyarrow`` and None. This is to help the execution engine
        use the most efficient way to execute the logic.
        """
    return None