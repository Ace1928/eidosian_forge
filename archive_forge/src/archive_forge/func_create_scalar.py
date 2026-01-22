from enum import Enum as PyEnum
import inspect
from functools import partial
from graphql import (
from ..utils.str_converters import to_camel_case
from ..utils.get_unbound_function import get_unbound_function
from .definitions import (
from .dynamic import Dynamic
from .enum import Enum
from .field import Field
from .inputobjecttype import InputObjectType
from .interface import Interface
from .objecttype import ObjectType
from .resolver import get_default_resolver
from .scalars import ID, Boolean, Float, Int, Scalar, String
from .structures import List, NonNull
from .union import Union
from .utils import get_field_as
@staticmethod
def create_scalar(graphene_type):
    _scalars = {String: GraphQLString, Int: GraphQLInt, Float: GraphQLFloat, Boolean: GraphQLBoolean, ID: GraphQLID}
    if graphene_type in _scalars:
        return _scalars[graphene_type]
    return GrapheneScalarType(graphene_type=graphene_type, name=graphene_type._meta.name, description=graphene_type._meta.description, serialize=getattr(graphene_type, 'serialize', None), parse_value=getattr(graphene_type, 'parse_value', None), parse_literal=getattr(graphene_type, 'parse_literal', None))