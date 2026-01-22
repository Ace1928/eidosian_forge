import datetime
import graphene
from graphene import relay
from graphene.types.resolver import dict_resolver
from ..deduplicator import deflate
def resolve_movie(event, info):
    return TEST_DATA['movies'][event['movie']]