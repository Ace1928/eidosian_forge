from typing import Union
from ..error import GraphQLError
from ..language import (
from ..type import GraphQLObjectType, GraphQLSchema
Extract the root type of the operation from the schema.

    .. deprecated:: 3.2
       Please use `GraphQLSchema.getRootType` instead. Will be removed in v3.3.
    