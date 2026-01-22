import functools
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
def get_or_register_sqlite_connection(model: 'SQLiteModelMixin', conn: Optional[Union['sqlite3.Connection', 'aiosqlite.Connection']]=None) -> Union['sqlite3.Connection', 'aiosqlite.Connection']:
    """
    Registers the SQLite Connection
    """
    global _sqlite_model_name_to_connection
    model_name = f'{model.__module__}.{model.__name__}'
    if conn and model_name not in _sqlite_model_name_to_connection:
        _sqlite_model_name_to_connection[model_name] = conn
    if model_name not in _sqlite_model_name_to_connection:
        raise ValueError(f'Model {model_name} not registered and no connection provided')
    return _sqlite_model_name_to_connection[model_name]