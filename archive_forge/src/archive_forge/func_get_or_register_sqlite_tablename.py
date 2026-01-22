import functools
from pydantic import BaseModel, computed_field
from typing import Any, Dict, List, Optional, Type, Union, TYPE_CHECKING
def get_or_register_sqlite_tablename(model: 'SQLiteModelMixin', tablename: Optional[str]=None) -> str:
    """
    Registers the SQLite Tablename
    """
    global _sqlite_model_name_to_tablename
    model_name = f'{model.__module__}.{model.__name__}'
    if not tablename and model_name not in _sqlite_model_name_to_tablename:
        if hasattr(model, 'tablename') and model.model_fields['tablename'].default is not None:
            tablename = model.model_fields['tablename'].default
        else:
            raise ValueError(f'Model {model_name} not registered and no tablename provided')
    if tablename and model_name not in _sqlite_model_name_to_tablename:
        _sqlite_model_name_to_tablename[model_name] = tablename
    if model_name not in _sqlite_model_name_to_tablename:
        raise ValueError(f'Model {model_name} not registered and no tablename provided')
    return _sqlite_model_name_to_tablename[model_name]