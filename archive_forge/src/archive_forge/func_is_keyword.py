import re
from sqlparse import tokens
def is_keyword(value):
    val = value.upper()
    return (KEYWORDS_COMMON.get(val) or KEYWORDS_ORACLE.get(val) or KEYWORDS_PLPGSQL.get(val) or KEYWORDS.get(val, tokens.Name), value)