from typing import cast, Union
from ..language import DocumentNode, Source, parse
from ..type import (
from .extend_schema import extend_schema_impl
def build_schema(source: Union[str, Source], assume_valid: bool=False, assume_valid_sdl: bool=False, no_location: bool=False, allow_legacy_fragment_variables: bool=False) -> GraphQLSchema:
    """Build a GraphQLSchema directly from a source document."""
    return build_ast_schema(parse(source, no_location=no_location, allow_legacy_fragment_variables=allow_legacy_fragment_variables), assume_valid=assume_valid, assume_valid_sdl=assume_valid_sdl)