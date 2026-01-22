import unittest
from numba import jit, njit, objmode, typeof, literally
from numba.extending import overload
from numba.core import types
from numba.core.errors import UnsupportedError
from numba.tests.support import (
@njit
def sum_jit_func(arg0=0, arg1=0, arg2=0, arg3=0, arg4=0, arg5=0, arg6=0, arg7=0, arg8=0, arg9=0, arg10=0, arg11=0, arg12=0, arg13=0, arg14=0, arg15=0, arg16=0, arg17=0, arg18=0, arg19=0, arg20=0, arg21=0, arg22=0, arg23=0, arg24=0, arg25=0, arg26=0, arg27=0, arg28=0, arg29=0, arg30=0, arg31=0, arg32=0, arg33=0, arg34=0, arg35=0, arg36=0, arg37=0, arg38=0, arg39=0, arg40=0, arg41=0, arg42=0, arg43=0, arg44=0, arg45=0, arg46=0):
    return arg0 + arg1 + arg2 + arg3 + arg4 + arg5 + arg6 + arg7 + arg8 + arg9 + arg10 + arg11 + arg12 + arg13 + arg14 + arg15 + arg16 + arg17 + arg18 + arg19 + arg20 + arg21 + arg22 + arg23 + arg24 + arg25 + arg26 + arg27 + arg28 + arg29 + arg30 + arg31 + arg32 + arg33 + arg34 + arg35 + arg36 + arg37 + arg38 + arg39 + arg40 + arg41 + arg42 + arg43 + arg44 + arg45 + arg46