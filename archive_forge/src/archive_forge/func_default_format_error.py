from graphql.error import GraphQLError
from graphene.types.schema import Schema
def default_format_error(error):
    if isinstance(error, GraphQLError):
        return error.formatted
    return {'message': str(error)}