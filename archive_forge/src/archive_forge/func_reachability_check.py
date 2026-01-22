from typing import Dict, Optional
from jedi.parser_utils import get_flow_branch_keyword, is_scope, get_parent_scope
from jedi.inference.recursion import execution_allowed
from jedi.inference.helpers import is_big_annoying_library
def reachability_check(context, value_scope, node, origin_scope=None):
    if is_big_annoying_library(context) or not context.inference_state.flow_analysis_enabled:
        return UNSURE
    first_flow_scope = get_parent_scope(node, include_flows=True)
    if origin_scope is not None:
        origin_flow_scopes = list(_get_flow_scopes(origin_scope))
        node_flow_scopes = list(_get_flow_scopes(node))
        branch_matches = True
        for flow_scope in origin_flow_scopes:
            if flow_scope in node_flow_scopes:
                node_keyword = get_flow_branch_keyword(flow_scope, node)
                origin_keyword = get_flow_branch_keyword(flow_scope, origin_scope)
                branch_matches = node_keyword == origin_keyword
                if flow_scope.type == 'if_stmt':
                    if not branch_matches:
                        return UNREACHABLE
                elif flow_scope.type == 'try_stmt':
                    if not branch_matches and origin_keyword == 'else' and (node_keyword == 'except'):
                        return UNREACHABLE
                if branch_matches:
                    break
        while origin_scope is not None:
            if first_flow_scope == origin_scope and branch_matches:
                return REACHABLE
            origin_scope = origin_scope.parent
    return _break_check(context, value_scope, first_flow_scope, node)