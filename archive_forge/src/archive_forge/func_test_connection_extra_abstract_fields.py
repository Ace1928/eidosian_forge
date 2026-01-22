import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_connection_extra_abstract_fields():

    class ConnectionWithNodes(Connection):

        class Meta:
            abstract = True

        @classmethod
        def __init_subclass_with_meta__(cls, node=None, name=None, **options):
            _meta = ConnectionOptions(cls)
            _meta.fields = {'nodes': Field(NonNull(List(node)), description='Contains all the nodes in this connection.')}
            return super(ConnectionWithNodes, cls).__init_subclass_with_meta__(node=node, name=name, _meta=_meta, **options)

    class MyObjectConnection(ConnectionWithNodes):

        class Meta:
            node = MyObject

        class Edge:
            other = String()
    assert MyObjectConnection._meta.name == 'MyObjectConnection'
    fields = MyObjectConnection._meta.fields
    assert list(fields) == ['nodes', 'page_info', 'edges']
    edge_field = fields['edges']
    pageinfo_field = fields['page_info']
    nodes_field = fields['nodes']
    assert isinstance(edge_field, Field)
    assert isinstance(edge_field.type, NonNull)
    assert isinstance(edge_field.type.of_type, List)
    assert edge_field.type.of_type.of_type == MyObjectConnection.Edge
    assert isinstance(pageinfo_field, Field)
    assert isinstance(pageinfo_field.type, NonNull)
    assert pageinfo_field.type.of_type == PageInfo
    assert isinstance(nodes_field, Field)
    assert isinstance(nodes_field.type, NonNull)
    assert isinstance(nodes_field.type.of_type, List)
    assert nodes_field.type.of_type.of_type == MyObject