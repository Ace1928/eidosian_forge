def atom_add_entry_type(node, _state):
    """
    @param node: the current node that could be modified
    @param state: current state
    @type state: L{Execution context<pyRdfa.state.ExecutionContext>}
    """

    def res_set(node):
        return True in [node.hasAttribute(a) for a in ['resource', 'about', 'href', 'src']]
    if node.tagName == 'entry' and (not res_set(node)) and (node.hasAttribute('typeof') == False):
        node.setAttribute('typeof', '')