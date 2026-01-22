import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_fetch_some_id_query(client):
    query = gql('\n        query FetchSomeIDQuery($someId: String!) {\n          human(id: $someId) {\n            name\n          }\n        }\n    ')
    params = {'someId': '1000'}
    expected = {'human': {'name': 'Luke Skywalker'}}
    result = client.execute(query, variable_values=params)
    assert result == expected