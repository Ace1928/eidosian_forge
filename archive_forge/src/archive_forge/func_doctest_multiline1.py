def doctest_multiline1():
    """The ipdoctest machinery must handle multiline examples gracefully.

    In [2]: for i in range(4):
       ...:     print(i)
       ...:      
    0
    1
    2
    3
    """