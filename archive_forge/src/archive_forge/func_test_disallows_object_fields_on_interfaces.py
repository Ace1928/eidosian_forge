import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
def test_disallows_object_fields_on_interfaces(client):
    query = '\n        query DroidFieldOnCharacter {\n          hero {\n            name\n            primaryFunction\n          }\n        }\n    '
    assert validation_errors(client, query)