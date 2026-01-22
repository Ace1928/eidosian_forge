from types import GenericAlias
def get_ready(self):
    """Return a tuple of all the nodes that are ready.

        Initially it returns all nodes with no predecessors; once those are marked
        as processed by calling "done", further calls will return all new nodes that
        have all their predecessors already processed. Once no more progress can be made,
        empty tuples are returned.

        Raises ValueError if called without calling "prepare" previously.
        """
    if self._ready_nodes is None:
        raise ValueError('prepare() must be called first')
    result = tuple(self._ready_nodes)
    n2i = self._node2info
    for node in result:
        n2i[node].npredecessors = _NODE_OUT
    self._ready_nodes.clear()
    self._npassedout += len(result)
    return result