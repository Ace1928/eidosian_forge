import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
@pytest.fixture
def local_schema():
    return Client(schema=StarWarsSchema)