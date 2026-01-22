import datetime
import graphene
from graphene import relay
from graphene.types.resolver import dict_resolver
from ..deduplicator import deflate
def test_example_end_to_end():

    class Movie(graphene.ObjectType):

        class Meta:
            interfaces = (relay.Node,)
            default_resolver = dict_resolver
        name = graphene.String(required=True)
        synopsis = graphene.String(required=True)

    class Event(graphene.ObjectType):

        class Meta:
            interfaces = (relay.Node,)
            default_resolver = dict_resolver
        movie = graphene.Field(Movie, required=True)
        date = graphene.types.datetime.Date(required=True)

        def resolve_movie(event, info):
            return TEST_DATA['movies'][event['movie']]

    class Query(graphene.ObjectType):
        events = graphene.List(graphene.NonNull(Event), required=True)

        def resolve_events(_, info):
            return TEST_DATA['events']
    schema = graphene.Schema(query=Query)
    query = '        {\n            events {\n                __typename\n                id\n                date\n                movie {\n                    __typename\n                    id\n                    name\n                    synopsis\n                }\n            }\n        }\n    '
    result = schema.execute(query)
    assert not result.errors
    data = deflate(result.data)
    assert data == {'events': [{'__typename': 'Event', 'id': 'RXZlbnQ6NTY4', 'date': '2017-05-19', 'movie': {'__typename': 'Movie', 'id': 'TW92aWU6MTE5ODM1OQ==', 'name': 'King Arthur: Legend of the Sword', 'synopsis': "When the child Arthur's father is murdered, Vortigern, Arthur's uncle, seizes the crown. Robbed of his birthright and with no idea who he truly is..."}}, {'__typename': 'Event', 'id': 'RXZlbnQ6MjM0', 'date': '2017-05-20', 'movie': {'__typename': 'Movie', 'id': 'TW92aWU6MTE5ODM1OQ=='}}]}