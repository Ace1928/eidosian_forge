from __future__ import annotations
import abc
import typing as t
from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth
from sqlglot.trie import TrieResult, in_trie, new_trie
def ensure_column_mapping(mapping: t.Optional[ColumnMapping]) -> t.Dict:
    if mapping is None:
        return {}
    elif isinstance(mapping, dict):
        return mapping
    elif isinstance(mapping, str):
        col_name_type_strs = [x.strip() for x in mapping.split(',')]
        return {name_type_str.split(':')[0].strip(): name_type_str.split(':')[1].strip() for name_type_str in col_name_type_strs}
    elif hasattr(mapping, 'simpleString'):
        return {struct_field.name: struct_field.dataType.simpleString() for struct_field in mapping}
    elif isinstance(mapping, list):
        return {x.strip(): None for x in mapping}
    raise ValueError(f'Invalid mapping provided: {type(mapping)}')