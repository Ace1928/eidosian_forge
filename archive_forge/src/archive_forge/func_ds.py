import pytest
from gql import Client
from gql.dsl import DSLSchema
from .schema import StarWarsSchema
@pytest.fixture
def ds():
    client = Client(schema=StarWarsSchema)
    ds = DSLSchema(client)
    return ds