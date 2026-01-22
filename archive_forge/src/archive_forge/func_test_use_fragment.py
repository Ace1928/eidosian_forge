import pytest
from graphql.error import format_error
from gql import Client, gql
from .schema import StarWarsSchema
def test_use_fragment(client):
    query = gql('\n        query UseFragment {\n          luke: human(id: "1000") {\n            ...HumanFragment\n          }\n          leia: human(id: "1003") {\n            ...HumanFragment\n          }\n        }\n        fragment HumanFragment on Human {\n          name\n          homePlanet\n        }\n    ')
    expected = {'luke': {'name': 'Luke Skywalker', 'homePlanet': 'Tatooine'}, 'leia': {'name': 'Leia Organa', 'homePlanet': 'Alderaan'}}
    result = client.execute(query)
    assert result == expected