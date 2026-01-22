import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_hero_name_and_friends_query(client):
    query = gql('\n        query HeroNameAndFriendsQuery {\n          hero {\n            id\n            name\n            friends {\n              name\n            }\n          }\n        }\n    ')
    expected = {'hero': {'id': '2001', 'name': 'R2-D2', 'friends': [{'name': 'Luke Skywalker'}, {'name': 'Han Solo'}, {'name': 'Leia Organa'}]}}
    result = client.execute(query)
    assert result == expected