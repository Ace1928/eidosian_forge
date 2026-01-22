import re
from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
Returns either a whitespace or the line breaks from token.