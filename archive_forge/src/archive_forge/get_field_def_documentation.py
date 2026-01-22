from ..type.definition import (GraphQLInterfaceType, GraphQLObjectType,
from ..type.introspection import (SchemaMetaFieldDef, TypeMetaFieldDef,
Not exactly the same as the executor's definition of get_field_def, in this
    statically evaluated environment we do not always have an Object type,
    and need to handle Interface and Union types.