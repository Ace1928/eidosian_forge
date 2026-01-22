import functools
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
def get_sqlite_model_config(model: 'SQLiteModelMixin', tablename: Optional[str]=None) -> SQLiteModelConfig:
    """
    Returns the SQLite Model Config
    """
    model_name = f'{model.__module__}.{model.__name__}'
    if model_name not in _sqlite_model_config_registry:
        if tablename and tablename in _sqlite_model_schema_registry:
            schema = _sqlite_model_schema_registry[tablename]
        else:
            schema = retrieve_sqlite_model_schema(model_name)
        config = SQLiteModelConfig.model_validate(schema)
        config.sql_model_name = model_name
        if model_name in _sqlite_model_name_to_connection:
            config.db_conn = _sqlite_model_name_to_connection[model_name]
        _sqlite_model_config_registry[model_name] = config
    return _sqlite_model_config_registry[model_name]