import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_check_type_of_r2(client):
    query = gql('\n        query CheckTypeOfR2 {\n          hero {\n            __typename\n            name\n          }\n        }\n    ')
    expected = {'hero': {'__typename': 'Droid', 'name': 'R2-D2'}}
    result = client.execute(query)
    assert result == expected