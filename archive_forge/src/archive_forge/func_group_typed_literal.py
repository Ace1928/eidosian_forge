from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_typed_literal(tlist):

    def match(token):
        return imt(token, m=sql.TypedLiteral.M_OPEN)

    def match_to_extend(token):
        return isinstance(token, sql.TypedLiteral)

    def valid_prev(token):
        return token is not None

    def valid_next(token):
        return token is not None and token.match(*sql.TypedLiteral.M_CLOSE)

    def valid_final(token):
        return token is not None and token.match(*sql.TypedLiteral.M_EXTEND)

    def post(tlist, pidx, tidx, nidx):
        return (tidx, nidx)
    _group(tlist, sql.TypedLiteral, match, valid_prev, valid_next, post, extend=False)
    _group(tlist, sql.TypedLiteral, match_to_extend, valid_prev, valid_final, post, extend=True)