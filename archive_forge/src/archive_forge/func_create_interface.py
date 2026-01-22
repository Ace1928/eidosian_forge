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
def create_interface(self, graphene_type):
    resolve_type = partial(self.resolve_type, graphene_type.resolve_type, graphene_type._meta.name) if graphene_type.resolve_type else None

    def interfaces():
        interfaces = []
        for graphene_interface in graphene_type._meta.interfaces:
            interface = self.add_type(graphene_interface)
            assert interface.graphene_type == graphene_interface
            interfaces.append(interface)
        return interfaces
    return GrapheneInterfaceType(graphene_type=graphene_type, name=graphene_type._meta.name, description=graphene_type._meta.description, fields=partial(self.create_fields_for_type, graphene_type), interfaces=interfaces, resolve_type=resolve_type)