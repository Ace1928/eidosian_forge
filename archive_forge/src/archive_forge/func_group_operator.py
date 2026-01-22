from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_operator(tlist):
    ttypes = T_NUMERICAL + T_STRING + T_NAME
    sqlcls = (sql.SquareBrackets, sql.Parenthesis, sql.Function, sql.Identifier, sql.Operation)

    def match(token):
        return imt(token, t=(T.Operator, T.Wildcard))

    def valid(token):
        return imt(token, i=sqlcls, t=ttypes)

    def post(tlist, pidx, tidx, nidx):
        tlist[tidx].ttype = T.Operator
        return (pidx, nidx)
    valid_prev = valid_next = valid
    _group(tlist, sql.Operation, match, valid_prev, valid_next, post, extend=False)