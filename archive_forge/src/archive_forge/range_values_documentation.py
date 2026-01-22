import gast as ast
from collections import defaultdict
from functools import reduce
from pythran.analyses import Aliases, CFG
from pythran.intrinsic import Intrinsic
from pythran.passmanager import ModuleAnalysis
from pythran.interval import Interval, IntervalTuple, UNKNOWN_RANGE
from pythran.tables import MODULES, attributes
 Handle iterate variable across branches

        >>> import gast as ast
        >>> from pythran import passmanager, backend
        >>> pm = passmanager.PassManager("test")

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 1: b = 1
        ...     else: b = 3
        ...     pass''')

        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 1: b = a
        ...     else: b = 3
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=2, high=inf)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if 0 < a < 4: b = a
        ...     else: b = 3
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if (0 < a) and (a < 4): b = a
        ...     else: b = 3
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if (a == 1) or (a == 2): b = a
        ...     else: b = 3
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=1, high=3)

        >>> node = ast.parse('''
        ... def foo(a):
        ...     b = 5
        ...     if a > 0: b = a
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['a'], res['b']
        (Interval(low=-inf, high=inf), Interval(low=1, high=inf))

        >>> node = ast.parse('''
        ... def foo(a):
        ...     if a > 3: b = 1
        ...     else: b = 2
        ...     if a > 1: b = 2
        ...     pass''')
        >>> res = pm.gather(RangeValues, node)
        >>> res['b']
        Interval(low=2, high=2)
        