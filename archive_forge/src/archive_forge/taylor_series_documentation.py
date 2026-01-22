from pyomo.core.expr import identify_variables, value, differentiate
import logging
import math

    This last bit of code is just for higher order taylor series expansions.
    The recursive function _loop modifies derivs in place so that derivs becomes a 
    list of lists of lists... However, _loop is also a generator so that 
    we don't have to loop through it twice. _loop yields two lists. The 
    first is a list of indices corresponding to the first k-1 variables that
    differentiation is being done with respect to. The second is a list of 
    derivatives. Each entry in this list is the derivative with respect to 
    the first k-1 variables and the kth variable, whose index matches the 
    index in _derivs.
    