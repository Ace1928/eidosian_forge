import json
from ..error import GraphQLSyntaxError
def get_token_desc(token):
    if token.value:
        return u'{} "{}"'.format(get_token_kind_desc(token.kind), token.value)
    else:
        return get_token_kind_desc(token.kind)