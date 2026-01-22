import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_hero_name_query(client):
    query = gql('\n        query HeroNameQuery {\n          hero {\n            name\n          }\n        }\n    ')
    expected = {'hero': {'name': 'R2-D2'}}
    result = client.execute(query)
    assert result == expected