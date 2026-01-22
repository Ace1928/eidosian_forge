from __future__ import absolute_import
import ast
from sentry_sdk import Hub, serializer
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.utils import walk_exception_chain, iter_stacks
def pure_eval_frame(frame):
    source = executing.Source.for_frame(frame)
    if not source.tree:
        return {}
    statements = source.statements_at_line(frame.f_lineno)
    if not statements:
        return {}
    scope = stmt = list(statements)[0]
    while True:
        scope = scope.parent
        if isinstance(scope, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            break
    evaluator = pure_eval.Evaluator.from_frame(frame)
    expressions = evaluator.interesting_expressions_grouped(scope)

    def closeness(expression):
        nodes, _value = expression

        def start(n):
            return (n.lineno, n.col_offset)
        nodes_before_stmt = [node for node in nodes if start(node) < stmt.last_token.end]
        if nodes_before_stmt:
            return max((start(node) for node in nodes_before_stmt))
        else:
            lineno, col_offset = min((start(node) for node in nodes))
            return (-lineno, -col_offset)
    atok = source.asttokens()
    expressions.sort(key=closeness, reverse=True)
    return {atok.get_text(nodes[0]): value for nodes, value in expressions[:serializer.MAX_DATABAG_BREADTH]}