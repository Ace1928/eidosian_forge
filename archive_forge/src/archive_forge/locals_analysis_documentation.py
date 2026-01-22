from pythran.passmanager import ModuleAnalysis
import pythran.metadata as md
import gast as ast

    Statically compute the value of locals() before each statement

    Yields a dictionary binding every node to the set of variable names defined
    *before* this node.

    Following snippet illustrates its behavior:
    >>> import gast as ast
    >>> from pythran import passmanager
    >>> pm = passmanager.PassManager('test')
    >>> code = '''
    ... def b(n):
    ...     m = n + 1
    ...     def b(n):
    ...         return n + 1
    ...     return b(m)'''
    >>> tree = ast.parse(code)
    >>> l = pm.gather(Locals, tree)
    >>> sorted(l[tree.body[0].body[0]])
    ['n']
    >>> sorted(l[tree.body[0].body[1]])
    ['b', 'm', 'n']
    