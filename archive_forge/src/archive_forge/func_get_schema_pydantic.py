import functools
import inspect
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, cast
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel
@get_schema.register(type(BaseModel))
def get_schema_pydantic(model: Type[BaseModel]):
    """Return the schema of a Pydantic model."""
    if not type(model) == type(BaseModel):
        raise TypeError('The `schema` filter only applies to Pydantic models.')
    if hasattr(model, 'model_json_schema'):
        def_key = '$defs'
        raw_schema = model.model_json_schema()
    else:
        def_key = 'definitions'
        raw_schema = model.schema()
    definitions = raw_schema.get(def_key, None)
    schema = parse_pydantic_schema(raw_schema, definitions)
    return json.dumps(schema, indent=2)