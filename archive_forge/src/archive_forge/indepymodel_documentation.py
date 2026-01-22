from pydantic import BaseModel, create_model, ValidationError, Field
from typing import Dict, Type, Any, Tuple, Union, TypeVar, get_type_hints, Optional

        Dynamically creates a new model extending the current one with additional fields.

        Args:
            name (str): The name of the new model class.
            fields (FieldsDictType): A dictionary where keys are field names and values are tuples of (type, default value).

        Returns:
            ModelType: A new Pydantic model class with the specified fields added.

        Raises:
            ValueError: If any of the new fields clash with existing fields in the base model.
        