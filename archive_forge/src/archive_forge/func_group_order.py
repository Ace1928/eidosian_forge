from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_order(tlist):
    """Group together Identifier and Asc/Desc token"""
    tidx, token = tlist.token_next_by(t=T.Keyword.Order)
    while token:
        pidx, prev_ = tlist.token_prev(tidx)
        if imt(prev_, i=sql.Identifier, t=T.Number):
            tlist.group_tokens(sql.Identifier, pidx, tidx)
            tidx = pidx
        tidx, token = tlist.token_next_by(t=T.Keyword.Order, idx=tidx)