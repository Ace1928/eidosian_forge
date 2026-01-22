import datetime
import graphene
from graphql.utilities import print_schema
def test_schema_printable_with_default_datetime_value():
    schema = graphene.Schema(query=Query, mutation=Mutations)
    schema_str = print_schema(schema.graphql_schema)
    assert schema_str, 'empty schema printed'