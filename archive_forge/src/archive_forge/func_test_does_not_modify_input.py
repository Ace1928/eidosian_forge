import datetime
import graphene
from graphene import relay
from graphene.types.resolver import dict_resolver
from ..deduplicator import deflate
def test_does_not_modify_input():
    response = {'data': [{'__typename': 'foo', 'id': 1, 'name': 'foo'}, {'__typename': 'foo', 'id': 1, 'name': 'foo'}]}
    deflate(response)
    assert response == {'data': [{'__typename': 'foo', 'id': 1, 'name': 'foo'}, {'__typename': 'foo', 'id': 1, 'name': 'foo'}]}