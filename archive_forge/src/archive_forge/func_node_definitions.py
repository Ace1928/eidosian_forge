from typing import Any, Callable, NamedTuple, Optional, Union
from graphql_relay.utils.base64 import base64, unbase64
from graphql import (
def node_definitions(fetch_by_id: Callable[[str, GraphQLResolveInfo], Any], type_resolver: Optional[GraphQLTypeResolver]=None) -> GraphQLNodeDefinitions:
    """
    Given a function to map from an ID to an underlying object, and a function
    to map from an underlying object to the concrete GraphQLObjectType it
    corresponds to, constructs a `Node` interface that objects can implement,
    and a field object to be used as a `node` root field.

    If the type_resolver is omitted, object resolution on the interface will be
    handled with the `is_type_of` method on object types, as with any GraphQL
    interface without a provided `resolve_type` method.
    """
    node_interface = GraphQLInterfaceType('Node', description='An object with an ID', fields=lambda: {'id': GraphQLField(GraphQLNonNull(GraphQLID), description='The id of the object.')}, resolve_type=type_resolver)
    node_field = GraphQLField(node_interface, description='Fetches an object given its ID', args={'id': GraphQLArgument(GraphQLNonNull(GraphQLID), description='The ID of an object')}, resolve=lambda _obj, info, id: fetch_by_id(id, info))
    nodes_field = GraphQLField(GraphQLNonNull(GraphQLList(node_interface)), description='Fetches objects given their IDs', args={'ids': GraphQLArgument(GraphQLNonNull(GraphQLList(GraphQLNonNull(GraphQLID))), description='The IDs of objects')}, resolve=lambda _obj, info, ids: [fetch_by_id(id_, info) for id_ in ids])
    return GraphQLNodeDefinitions(node_interface, node_field, nodes_field)