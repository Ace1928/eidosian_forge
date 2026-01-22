import datetime
import graphene
from graphene import relay
from graphene.types.resolver import dict_resolver
from ..deduplicator import deflate
def test_does_not_modify_object_without_typename_and_id():
    response = {'foo': 'bar'}
    deflated_response = deflate(response)
    assert deflated_response == {'foo': 'bar'}