import stack_data
import sys
def doctest_tb_verbose():
    """
    In [5]: xmode verbose
    Exception reporting mode: Verbose

    In [6]: run simpleerr.py
    ---------------------------------------------------------------------------
    ZeroDivisionError                         Traceback (most recent call last)
    <BLANKLINE>
    ...
         30     except IndexError:
         31         mode = 'div'
    ---> 33     bar(mode)
            mode = 'div'
    <BLANKLINE>
    ... in bar(mode='div')
         15     "bar"
         16     if mode=='div':
    ---> 17         div0()
         18     elif mode=='exit':
         19         try:
    <BLANKLINE>
    ... in div0()
          6     x = 1
          7     y = 0
    ----> 8     x/y
            x = 1
            y = 0
    <BLANKLINE>
    ZeroDivisionError: ...
    """