from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def group_comparison(tlist):
    sqlcls = (sql.Parenthesis, sql.Function, sql.Identifier, sql.Operation)
    ttypes = T_NUMERICAL + T_STRING + T_NAME

    def match(token):
        return token.ttype == T.Operator.Comparison

    def valid(token):
        if imt(token, t=ttypes, i=sqlcls):
            return True
        elif token and token.is_keyword and (token.normalized == 'NULL'):
            return True
        else:
            return False

    def post(tlist, pidx, tidx, nidx):
        return (pidx, nidx)
    valid_prev = valid_next = valid
    _group(tlist, sql.Comparison, match, valid_prev, valid_next, post, extend=False)