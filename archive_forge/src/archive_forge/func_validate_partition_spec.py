from typing import Any, Callable, Dict
from triad import Schema
from triad.utils.assertion import assert_or_throw
from fugue._utils.interfaceless import parse_comment_annotation
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.exceptions import (
def validate_partition_spec(spec: PartitionSpec, rules: Dict[str, Any]) -> None:
    for k, v in rules.items():
        if k in ['partitionby_has', 'partitionby_is']:
            for x in v:
                assert_or_throw(x in spec.partition_by, lambda: FugueWorkflowCompileValidationError(f'required partition key {x} is not in {spec}'))
            if k == 'partitionby_is':
                assert_or_throw(len(v) == len(spec.partition_by), lambda: FugueWorkflowCompileValidationError(f'{v} does not match {spec}'))
        if k in ['presort_has', 'presort_is']:
            expected = spec.presort
            for pk, pv in v:
                o = 'ASC' if pv else 'DESC'
                assert_or_throw(pk in expected, lambda: FugueWorkflowCompileValidationError(f'required presort key {pk} is not in presort of {spec}'))
                assert_or_throw(pv == expected[pk], lambda: FugueWorkflowCompileValidationError(f"({pk},{o}) order does't match presort of {spec}"))
            if k == 'presort_is':
                assert_or_throw(len(v) == len(expected), lambda: FugueWorkflowCompileValidationError(f'{v} does not match {spec}'))
                assert_or_throw(v == list(expected.items()), lambda: FugueWorkflowCompileValidationError(f'{v} order does not match {spec}'))