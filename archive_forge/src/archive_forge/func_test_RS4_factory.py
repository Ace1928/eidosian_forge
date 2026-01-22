import pytest
import sys
import textwrap
import rpy2.robjects as robjects
import rpy2.robjects.methods as methods
def test_RS4_factory():
    rclassname = 'Foo'
    robjects.r['setClass'](rclassname, robjects.r('list(bar="numeric")'))
    obj = robjects.r['new'](rclassname)
    f_rs4i = methods.rs4instance_factory(obj)
    assert rclassname == type(f_rs4i).__name__