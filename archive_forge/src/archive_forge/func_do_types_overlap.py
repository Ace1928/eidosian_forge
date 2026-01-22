from ..type.definition import (GraphQLInterfaceType, GraphQLList,
def do_types_overlap(schema, t1, t2):
    if t1 == t2:
        return True
    if isinstance(t1, (GraphQLInterfaceType, GraphQLUnionType)):
        if isinstance(t2, (GraphQLInterfaceType, GraphQLUnionType)):
            s = any([schema.is_possible_type(t2, type) for type in schema.get_possible_types(t1)])
            return s
        r = schema.is_possible_type(t1, t2)
        return r
    if isinstance(t2, (GraphQLInterfaceType, GraphQLUnionType)):
        t = schema.is_possible_type(t2, t1)
        return t
    return False