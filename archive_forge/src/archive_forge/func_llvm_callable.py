import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
@doctest_depends_on(modules=('llvmlite', 'scipy'))
def llvm_callable(args, expr, callback_type=None):
    """Compile function from a SymPy expression

    Expressions are evaluated using double precision arithmetic.
    Some single argument math functions (exp, sin, cos, etc.) are supported
    in expressions.

    Parameters
    ==========

    args : List of Symbol
        Arguments to the generated function.  Usually the free symbols in
        the expression.  Currently each one is assumed to convert to
        a double precision scalar.
    expr : Expr, or (Replacements, Expr) as returned from 'cse'
        Expression to compile.
    callback_type : string
        Create function with signature appropriate to use as a callback.
        Currently supported:
           'scipy.integrate'
           'scipy.integrate.test'
           'cubature'

    Returns
    =======

    Compiled function that can evaluate the expression.

    Examples
    ========

    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy.abc import a
    >>> e = a*a + a + 1
    >>> e1 = jit.llvm_callable([a], e)
    >>> e.subs(a, 1.1)   # Evaluate via substitution
    3.31000000000000
    >>> e1(1.1)  # Evaluate using JIT-compiled code
    3.3100000000000005


    Callbacks for integration functions can be JIT compiled.
    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy.abc import a
    >>> from sympy import integrate
    >>> from scipy.integrate import quad
    >>> e = a*a
    >>> e1 = jit.llvm_callable([a], e, callback_type='scipy.integrate')
    >>> integrate(e, (a, 0.0, 2.0))
    2.66666666666667
    >>> quad(e1, 0.0, 2.0)[0]
    2.66666666666667

    The 'cubature' callback is for the Python wrapper around the
    cubature package ( https://github.com/saullocastro/cubature )
    and ( http://ab-initio.mit.edu/wiki/index.php/Cubature )

    There are two signatures for the SciPy integration callbacks.
    The first ('scipy.integrate') is the function to be passed to the
    integration routine, and will pass the signature checks.
    The second ('scipy.integrate.test') is only useful for directly calling
    the function using ctypes variables. It will not pass the signature checks
    for scipy.integrate.

    The return value from the cse module can also be compiled.  This
    can improve the performance of the compiled function.  If multiple
    expressions are given to cse, the compiled function returns a tuple.
    The 'cubature' callback handles multiple expressions (set `fdim`
    to match in the integration call.)
    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy import cse
    >>> from sympy.abc import x,y
    >>> e1 = x*x + y*y
    >>> e2 = 4*(x*x + y*y) + 8.0
    >>> after_cse = cse([e1,e2])
    >>> after_cse
    ([(x0, x**2), (x1, y**2)], [x0 + x1, 4*x0 + 4*x1 + 8.0])
    >>> j1 = jit.llvm_callable([x,y], after_cse)
    >>> j1(1.0, 2.0)
    (5.0, 28.0)
    """
    if not llvmlite:
        raise ImportError('llvmlite is required for llvmjitcode')
    signature = CodeSignature(ctypes.py_object)
    arg_ctypes = []
    if callback_type is None:
        for _ in args:
            arg_ctype = ctypes.c_double
            arg_ctypes.append(arg_ctype)
    elif callback_type in ('scipy.integrate', 'scipy.integrate.test'):
        signature.ret_type = ctypes.c_double
        arg_ctypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        arg_ctypes_formal = [ctypes.c_int, ctypes.c_double]
        signature.input_arg = 1
    elif callback_type == 'cubature':
        arg_ctypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        signature.ret_type = ctypes.c_int
        signature.input_arg = 1
        signature.ret_arg = 4
    else:
        raise ValueError('Unknown callback type: %s' % callback_type)
    signature.arg_ctypes = arg_ctypes
    fptr = _llvm_jit_code(args, expr, signature, callback_type)
    if callback_type and callback_type == 'scipy.integrate':
        arg_ctypes = arg_ctypes_formal
    if signature.ret_type == ctypes.py_object:
        FUNCTYPE = ctypes.PYFUNCTYPE
    else:
        FUNCTYPE = ctypes.CFUNCTYPE
    cfunc = FUNCTYPE(signature.ret_type, *arg_ctypes)(fptr)
    return cfunc