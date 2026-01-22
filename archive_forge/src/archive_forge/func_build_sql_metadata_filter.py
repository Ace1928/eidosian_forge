from __future__ import annotations
import datetime
from typing import Any, Dict, List, Optional, Union
def build_sql_metadata_filter(conditional: Optional[str]='AND', metadata_key: Optional[str]='_metadata', **filters: Dict[str, Union[int, float, datetime.datetime, Dict, List, Any]]) -> str:
    """
    Constructs the WHERE clause for the `_metadata` property because it's a `jsonb` field
    """
    q = ''
    for key, value in filters.items():
        if 'date' in key:
            if isinstance(value, list):
                if isinstance(value[0], datetime.datetime):
                    value[0] = value[0].strftime('%Y-%m-%d %H:%M:%S')
                    value[1] = value[1].strftime('%Y-%m-%d %H:%M:%S')
                q += f"jsonb_path_exists({metadata_key}, '$.{key} ? (@ BETWEEN {value[0]} AND {value[1]} || @ == null)' {conditional} "
            else:
                op = '>' if key == 'open_date' else '<='
                q += f"jsonb_path_exists({metadata_key}, '$.{key} ? (@ {op} {value} || @ == null)' {conditional} "
            continue
        if isinstance(value, (int, float)):
            op = '>=' if 'min' in key else '<='
            q += f"jsonb_path_exists({metadata_key}, '$.{key} ? (@ {op} {value} || @ == null)' {conditional} "
            continue
        if isinstance(value, str):
            q += f"jsonb_path_exists({metadata_key}, '$.{key} ? (@ ILIKE %'{value}'% || @ == null)' {conditional} "
            continue
        if isinstance(value, list):
            q += f"jsonb_path_exists({metadata_key}, '$.{key} ? (@ && {value} || @ == null)' {conditional} "
            continue
        if isinstance(value, dict):
            q += f"jsonb_path_exists({metadata_key}, '$.{key} ? (@ ?& {value.keys()} || @ == null)' {conditional} "
            continue
    q = q[:-len(conditional) - 1]
    return q