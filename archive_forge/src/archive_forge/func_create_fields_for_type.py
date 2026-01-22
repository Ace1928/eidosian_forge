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
def create_fields_for_type(self, graphene_type, is_input_type=False):
    create_graphql_type = self.add_type
    fields = {}
    for name, field in graphene_type._meta.fields.items():
        if isinstance(field, Dynamic):
            field = get_field_as(field.get_type(self), _as=Field)
            if not field:
                continue
        field_type = create_graphql_type(field.type)
        if is_input_type:
            _field = GraphQLInputField(field_type, default_value=field.default_value, out_name=name, description=field.description, deprecation_reason=field.deprecation_reason)
        else:
            args = {}
            for arg_name, arg in field.args.items():
                arg_type = create_graphql_type(arg.type)
                processed_arg_name = arg.name or self.get_name(arg_name)
                args[processed_arg_name] = GraphQLArgument(arg_type, out_name=arg_name, description=arg.description, default_value=arg.default_value, deprecation_reason=arg.deprecation_reason)
            subscribe = field.wrap_subscribe(self.get_function_for_type(graphene_type, f'subscribe_{name}', name, field.default_value))
            if subscribe:
                field_default_resolver = identity_resolve
            elif issubclass(graphene_type, ObjectType):
                default_resolver = graphene_type._meta.default_resolver or get_default_resolver()
                field_default_resolver = partial(default_resolver, name, field.default_value)
            else:
                field_default_resolver = None
            resolve = field.wrap_resolve(self.get_function_for_type(graphene_type, f'resolve_{name}', name, field.default_value) or field_default_resolver)
            _field = GraphQLField(field_type, args=args, resolve=resolve, subscribe=subscribe, deprecation_reason=field.deprecation_reason, description=field.description)
        field_name = field.name or self.get_name(name)
        fields[field_name] = _field
    return fields