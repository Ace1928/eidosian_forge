import json
from ..error import GraphQLSyntaxError
def get_token_kind_desc(kind):
    return TOKEN_DESCRIPTION[kind]