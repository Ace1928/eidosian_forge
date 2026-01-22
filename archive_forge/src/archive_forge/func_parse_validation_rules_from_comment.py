from typing import Any, Callable, Dict
from triad import Schema
from triad.utils.assertion import assert_or_throw
from fugue._utils.interfaceless import parse_comment_annotation
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.exceptions import (
def parse_validation_rules_from_comment(func: Callable) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for key in ['partitionby_has', 'partitionby_is', 'presort_has', 'presort_is', 'input_has', 'input_is']:
        v = parse_comment_annotation(func, key)
        if v is None:
            continue
        assert_or_throw(v != '', lambda: SyntaxError(f"{key} can't be empty"))
        res[key] = v
    return to_validation_rules(res)