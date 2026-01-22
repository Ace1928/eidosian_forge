from sqlglot import exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import build_scope, find_in_scope
from sqlglot.optimizer.simplify import simplify

    If the predicates are in DNF form, we can only push down conditions that are in all blocks.
    Additionally, we can't remove predicates from their original form.
    