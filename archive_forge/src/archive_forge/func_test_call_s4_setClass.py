import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_call_s4_setClass():
    r_setClass = rinterface.globalenv.find('setClass')
    r_representation = rinterface.globalenv.find('representation')
    attrnumeric = rinterface.StrSexpVector(['numeric'])
    classname = rinterface.StrSexpVector(['Track'])
    classrepr = r_representation(x=attrnumeric, y=attrnumeric)
    r_setClass(classname, classrepr)