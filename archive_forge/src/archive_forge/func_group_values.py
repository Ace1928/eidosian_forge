from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_values(tlist):
    tidx, token = tlist.token_next_by(m=(T.Keyword, 'VALUES'))
    start_idx = tidx
    end_idx = -1
    while token:
        if isinstance(token, sql.Parenthesis):
            end_idx = tidx
        tidx, token = tlist.token_next(tidx)
    if end_idx != -1:
        tlist.group_tokens(sql.Values, start_idx, end_idx, extend=True)