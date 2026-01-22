import re
from urllib import parse as urllib_parse
import pyparsing as pp
def search_query_builder(query):
    parsed_query = expr.parseString(query)[0]
    return _parsed_query2dict(parsed_query)