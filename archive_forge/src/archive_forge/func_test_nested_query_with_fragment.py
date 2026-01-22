import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
def test_nested_query_with_fragment(client):
    query = '\n        query NestedQueryWithFragment {\n          hero {\n            ...NameAndAppearances\n            friends {\n              ...NameAndAppearances\n              friends {\n                ...NameAndAppearances\n              }\n            }\n          }\n        }\n        fragment NameAndAppearances on Character {\n          name\n          appearsIn\n        }\n    '
    assert not validation_errors(client, query)