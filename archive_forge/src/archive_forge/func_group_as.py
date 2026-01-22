from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_as(tlist):

    def match(token):
        return token.is_keyword and token.normalized == 'AS'

    def valid_prev(token):
        return token.normalized == 'NULL' or not token.is_keyword

    def valid_next(token):
        ttypes = (T.DML, T.DDL, T.CTE)
        return not imt(token, t=ttypes) and token is not None

    def post(tlist, pidx, tidx, nidx):
        return (pidx, nidx)
    _group(tlist, sql.Identifier, match, valid_prev, valid_next, post)