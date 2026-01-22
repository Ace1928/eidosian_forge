from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
def infer_alias(self) -> ColumnExpr:
    return self if self.output_name != '' else self.alias(self.col.infer_alias().output_name)