import json
from typing import Type, Union
from pydantic import BaseModel
from transformers import SPIECE_UNDERLINE, PreTrainedTokenizerBase
def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
    """Convert a JSON schema to a string.

    Parameters
    ----------
    json_schema
        The JSON schema.

    Returns
    -------
    str
        The JSON schema converted to a string.

    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(f'Cannot parse schema {json_schema}. The schema must be either ' + 'a Pydantic class, a dictionary or a string that contains the JSON ' + 'schema specification')
    return schema_str