import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
def p_binary(this, p):
    alias = this._aliases.get(p.slice[2].type)
    p[0] = expressions.BinaryOperator(p[2], p[1], p[3], alias)