import re
from pytest import raises
from ...types import Argument, Field, Int, List, NonNull, ObjectType, Schema, String
from ..connection import (
from ..node import Node
def test_connection_override_fields():

    class ConnectionWithNodes(Connection):

        class Meta:
            abstract = True

        @classmethod
        def __init_subclass_with_meta__(cls, node=None, name=None, **options):
            _meta = ConnectionOptions(cls)
            base_name = re.sub('Connection$', '', name or cls.__name__) or node._meta.name
            edge_class = get_edge_class(cls, node, base_name)
            _meta.fields = {'page_info': Field(NonNull(PageInfo, name='pageInfo', required=True, description='Pagination data for this connection.')), 'edges': Field(NonNull(List(NonNull(edge_class))), description='Contains the nodes in this connection.')}
            return super(ConnectionWithNodes, cls).__init_subclass_with_meta__(node=node, name=name, _meta=_meta, **options)

    class MyObjectConnection(ConnectionWithNodes):

        class Meta:
            node = MyObject
    assert MyObjectConnection._meta.name == 'MyObjectConnection'
    fields = MyObjectConnection._meta.fields
    assert list(fields) == ['page_info', 'edges']
    edge_field = fields['edges']
    pageinfo_field = fields['page_info']
    assert isinstance(edge_field, Field)
    assert isinstance(edge_field.type, NonNull)
    assert isinstance(edge_field.type.of_type, List)
    assert isinstance(edge_field.type.of_type.of_type, NonNull)
    assert edge_field.type.of_type.of_type.of_type.__name__ == 'MyObjectEdge'
    assert isinstance(pageinfo_field, Field)
    assert isinstance(edge_field.type, NonNull)
    assert pageinfo_field.type.of_type == PageInfo