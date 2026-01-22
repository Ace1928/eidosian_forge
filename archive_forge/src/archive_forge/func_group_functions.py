from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
@recurse(sql.Function)
def group_functions(tlist):
    has_create = False
    has_table = False
    for tmp_token in tlist.tokens:
        if tmp_token.value == 'CREATE':
            has_create = True
        if tmp_token.value == 'TABLE':
            has_table = True
    if has_create and has_table:
        return
    tidx, token = tlist.token_next_by(t=T.Name)
    while token:
        nidx, next_ = tlist.token_next(tidx)
        if isinstance(next_, sql.Parenthesis):
            tlist.group_tokens(sql.Function, tidx, nidx)
        tidx, token = tlist.token_next_by(t=T.Name, idx=tidx)