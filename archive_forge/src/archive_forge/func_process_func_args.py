import urllib.request
import sys
from typing import Tuple
def process_func_args(s, hip_sig, decl, hip_func):
    if 'const cuComplex*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<const hipComplex*>'
    elif 'const cuDoubleComplex*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<const hipDoubleComplex*>'
    elif 'cuComplex*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<hipComplex*>'
    elif 'cuDoubleComplex*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<hipDoubleComplex*>'
    elif 'cuComplex' in s:
        s = s.split()
        decl += '  hipComplex blah;\n'
        decl += f'  blah.x={s[-1][:-1]}.x;\n  blah.y={s[-1][:-1]}.y;\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cuDoubleComplex' in s:
        s = s.split()
        decl += '  hipDoubleComplex blah;\n'
        decl += f'  blah.x={s[-1][:-1]}.x;\n  blah.y={s[-1][:-1]}.y;\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cudaDataType*' in s:
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'reinterpret_cast<hipDataType*>'
    elif 'cudaDataType' in s:
        s = s.split()
        decl += '  hipDataType blah = convert_hipDatatype('
        decl += s[-1][:-1] + ');\n'
        arg = 'blah' + s[-1][-1]
        cast = ''
    elif 'cusparseOrder_t*' in s:
        s = s.split()
        decl += '  hipsparseOrder_t blah2 = '
        decl += 'convert_hipsparseOrder_t(*' + s[-1][:-1] + ');\n'
        arg = '&blah2' + s[-1][-1]
        cast = ''
    elif 'cusparseOrder_t' in s:
        s = s.split()
        decl += '  hipsparseOrder_t blah2 = '
        decl += 'convert_hipsparseOrder_t(' + s[-1][:-1] + ');\n'
        arg = 'blah2' + s[-1][-1]
        cast = ''
    elif 'const void*' in s and hip_func == 'hipsparseSpVV_bufferSize':
        s = s.split()
        arg = '(' + s[-1][:-1] + ')' + s[-1][-1]
        cast = 'const_cast<void*>'
    else:
        s = s.split()
        arg = s[-1]
        cast = ''
    hip_sig += cast + arg + ' '
    return (hip_sig, decl)