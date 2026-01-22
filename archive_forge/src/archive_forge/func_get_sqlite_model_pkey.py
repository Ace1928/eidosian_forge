import functools
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
def get_sqlite_model_pkey(model: 'SQLiteModelMixin', tablename: Optional[str]=None) -> str:
    """
    Returns the SQLite Model PKey
    """
    tablename = get_or_register_sqlite_tablename(model, tablename)
    return _sqlite_model_schema_registry[tablename]['sql_pkey']