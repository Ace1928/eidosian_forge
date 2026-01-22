from __future__ import annotations as _annotations
import typing
from typing import Any
import typing_extensions
def build_metadata_dict(*, js_functions: list[GetJsonSchemaFunction] | None=None, js_annotation_functions: list[GetJsonSchemaFunction] | None=None, js_prefer_positional_arguments: bool | None=None, typed_dict_cls: type[Any] | None=None, initial_metadata: Any | None=None) -> Any:
    """Builds a dict to use as the metadata field of a CoreSchema object in a manner that is consistent
    with the CoreMetadataHandler class.
    """
    if initial_metadata is not None and (not isinstance(initial_metadata, dict)):
        raise TypeError(f'CoreSchema metadata should be a dict; got {initial_metadata!r}.')
    metadata = CoreMetadata(pydantic_js_functions=js_functions or [], pydantic_js_annotation_functions=js_annotation_functions or [], pydantic_js_prefer_positional_arguments=js_prefer_positional_arguments, pydantic_typed_dict_cls=typed_dict_cls)
    metadata = {k: v for k, v in metadata.items() if v is not None}
    if initial_metadata is not None:
        metadata = {**initial_metadata, **metadata}
    return metadata