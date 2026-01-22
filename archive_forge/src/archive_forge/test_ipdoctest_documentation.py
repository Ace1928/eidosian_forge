Defining builtins._ should not break anything outside the doctest
    while also should be working as expected inside the doctest.

    In [1]: import builtins

    In [2]: builtins._ = 42

    In [3]: builtins._
    Out[3]: 42

    In [4]: _
    Out[4]: 42
    