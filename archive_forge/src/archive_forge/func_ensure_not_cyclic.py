from functools import total_ordering
from django.db.migrations.state import ProjectState
from .exceptions import CircularDependencyError, NodeNotFoundError
def ensure_not_cyclic(self):
    todo = set(self.nodes)
    while todo:
        node = todo.pop()
        stack = [node]
        while stack:
            top = stack[-1]
            for child in self.node_map[top].children:
                node = child.key
                if node in stack:
                    cycle = stack[stack.index(node):]
                    raise CircularDependencyError(', '.join(('%s.%s' % n for n in cycle)))
                if node in todo:
                    stack.append(node)
                    todo.remove(node)
                    break
            else:
                node = stack.pop()