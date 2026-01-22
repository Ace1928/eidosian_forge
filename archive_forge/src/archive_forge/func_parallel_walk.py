import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def parallel_walk(node, other):
    """Walks two ASTs in parallel.

  The two trees must have identical structure.

  Args:
    node: Union[ast.AST, Iterable[ast.AST]]
    other: Union[ast.AST, Iterable[ast.AST]]
  Yields:
    Tuple[ast.AST, ast.AST]
  Raises:
    ValueError: if the two trees don't have identical structure.
  """
    if isinstance(node, (list, tuple)):
        node_stack = list(node)
    else:
        node_stack = [node]
    if isinstance(other, (list, tuple)):
        other_stack = list(other)
    else:
        other_stack = [other]
    while node_stack and other_stack:
        assert len(node_stack) == len(other_stack)
        n = node_stack.pop()
        o = other_stack.pop()
        if not isinstance(n, (ast.AST, gast.AST, str)) and n is not None or (not isinstance(o, (ast.AST, gast.AST, str)) and n is not None) or n.__class__.__name__ != o.__class__.__name__:
            raise ValueError('inconsistent nodes: {} ({}) and {} ({})'.format(n, n.__class__.__name__, o, o.__class__.__name__))
        yield (n, o)
        if isinstance(n, str):
            assert isinstance(o, str), 'The check above should have ensured this'
            continue
        if n is None:
            assert o is None, 'The check above should have ensured this'
            continue
        for f in n._fields:
            n_child = getattr(n, f, None)
            o_child = getattr(o, f, None)
            if f.startswith('__') or n_child is None or o_child is None:
                continue
            if isinstance(n_child, (list, tuple)):
                if not isinstance(o_child, (list, tuple)) or len(n_child) != len(o_child):
                    raise ValueError('inconsistent values for field {}: {} and {}'.format(f, n_child, o_child))
                node_stack.extend(n_child)
                other_stack.extend(o_child)
            elif isinstance(n_child, (gast.AST, ast.AST)):
                node_stack.append(n_child)
                other_stack.append(o_child)
            elif n_child != o_child:
                raise ValueError('inconsistent values for field {}: {} and {}'.format(f, n_child, o_child))