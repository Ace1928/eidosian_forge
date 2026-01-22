from typing import Any, Callable, Dict
from triad import Schema
from triad.utils.assertion import assert_or_throw
from fugue._utils.interfaceless import parse_comment_annotation
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.exceptions import (
def load_namespace_extensions(obj: Any) -> None:
    if is_namespace_extension(obj):
        from fugue_contrib import load_namespace
        load_namespace(obj[0])