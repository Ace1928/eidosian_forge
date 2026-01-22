from typing import Any, Dict, List, Set
from ...error import GraphQLError
from ...language import FragmentDefinitionNode, FragmentSpreadNode, VisitorAction, SKIP
from . import ASTValidationContext, ASTValidationRule
No fragment cycles

    The graph of fragment spreads must not form any cycles including spreading itself.
    Otherwise an operation could infinitely spread or infinitely execute on cycles in
    the underlying data.

    See https://spec.graphql.org/draft/#sec-Fragment-spreads-must-not-form-cycles
    