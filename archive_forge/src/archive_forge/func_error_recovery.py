from parso.python import tree
from parso.python.token import PythonTokenTypes
from parso.parser import BaseParser
def error_recovery(self, token):
    tos_nodes = self.stack[-1].nodes
    if tos_nodes:
        last_leaf = tos_nodes[-1].get_last_leaf()
    else:
        last_leaf = None
    if self._start_nonterminal == 'file_input' and (token.type == PythonTokenTypes.ENDMARKER or (token.type == DEDENT and (not last_leaf.value.endswith('\n')) and (not last_leaf.value.endswith('\r')))):
        if self.stack[-1].dfa.from_rule == 'simple_stmt':
            try:
                plan = self.stack[-1].dfa.transitions[PythonTokenTypes.NEWLINE]
            except KeyError:
                pass
            else:
                if plan.next_dfa.is_final and (not plan.dfa_pushes):
                    self.stack[-1].dfa = plan.next_dfa
                    self._add_token(token)
                    return
    if not self._error_recovery:
        return super().error_recovery(token)

    def current_suite(stack):
        for until_index, stack_node in reversed(list(enumerate(stack))):
            if stack_node.nonterminal == 'file_input':
                break
            elif stack_node.nonterminal == 'suite':
                if len(stack_node.nodes) != 1:
                    break
        return until_index
    until_index = current_suite(self.stack)
    if self._stack_removal(until_index + 1):
        self._add_token(token)
    else:
        typ, value, start_pos, prefix = token
        if typ == INDENT:
            self._omit_dedent_list.append(self._indent_counter)
        error_leaf = tree.PythonErrorLeaf(typ.name, value, start_pos, prefix)
        self.stack[-1].nodes.append(error_leaf)
    tos = self.stack[-1]
    if tos.nonterminal == 'suite':
        try:
            tos.dfa = tos.dfa.arcs['stmt']
        except KeyError:
            pass