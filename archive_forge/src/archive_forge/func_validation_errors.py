import pytest
from graphql import graphql
from graphql.utils.introspection_query import introspection_query
from gql import Client, gql
from .schema import StarWarsSchema
def validation_errors(client, query):
    query = gql(query)
    try:
        client.validate(query)
        return False
    except Exception:
        return True