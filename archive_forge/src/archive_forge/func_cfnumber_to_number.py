from ctypes import *
from ctypes import util
from .runtime import send_message, ObjCInstance
from .cocoatypes import *
def cfnumber_to_number(cfnumber):
    """Convert CFNumber to python int or float."""
    numeric_type = cf.CFNumberGetType(cfnumber)
    cfnum_to_ctype = {kCFNumberSInt8Type: c_int8, kCFNumberSInt16Type: c_int16, kCFNumberSInt32Type: c_int32, kCFNumberSInt64Type: c_int64, kCFNumberFloat32Type: c_float, kCFNumberFloat64Type: c_double, kCFNumberCharType: c_byte, kCFNumberShortType: c_short, kCFNumberIntType: c_int, kCFNumberLongType: c_long, kCFNumberLongLongType: c_longlong, kCFNumberFloatType: c_float, kCFNumberDoubleType: c_double, kCFNumberCFIndexType: CFIndex, kCFNumberCGFloatType: CGFloat}
    if numeric_type in cfnum_to_ctype:
        t = cfnum_to_ctype[numeric_type]
        result = t()
        if cf.CFNumberGetValue(cfnumber, numeric_type, byref(result)):
            return result.value
    else:
        raise Exception('cfnumber_to_number: unhandled CFNumber type %d' % numeric_type)