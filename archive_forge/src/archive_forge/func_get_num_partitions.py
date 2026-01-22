import json
from typing import Any, Dict, List, Tuple
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_size
from triad.utils.hash import to_uuid
from triad.utils.pyarrow import SchemaedDataPartitioner
from triad.utils.schema import safe_split_out_of_quote, unquote_name
def get_num_partitions(self, **expr_map_funcs: Any) -> int:
    """Convert ``num_partitions`` expression to int number

        :param expr_map_funcs: lambda functions (no parameter) for keywords
        :return: integer value of the partitions

        .. admonition:: Examples

            >>> p = PartitionSpec(num="ROWCOUNT/2")
            >>> p.get_num_partitions(ROWCOUNT=lambda: df.count())
        """
    expr = self.num_partitions
    for k, v in expr_map_funcs.items():
        if k in expr:
            value = str(v())
            expr = expr.replace(k, value)
    return int(eval(expr))