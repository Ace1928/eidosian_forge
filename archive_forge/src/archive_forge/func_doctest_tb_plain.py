import stack_data
import sys
def doctest_tb_plain():
    """
    In [18]: xmode plain
    Exception reporting mode: Plain

    In [19]: run simpleerr.py
    Traceback (most recent call last):
      File ...:...
        bar(mode)
      File ...:... in bar
        div0()
      File ...:... in div0
        x/y
    ZeroDivisionError: ...
    """