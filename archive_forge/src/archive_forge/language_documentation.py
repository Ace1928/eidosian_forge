import typing
import uuid
from rpy2.robjects import conversion
from rpy2.robjects.robject import RObject
import rpy2.rinterface as ri
R language object.

    R language objects are unevaluated constructs using the R language.
    They can be found in the default values for named arguments, for example:
    ```r
    r_function(x, n = ncol(x))
    ```
    The default value for `n` is then the result of calling the R function
    `ncol()` on the object `x` passed at the first argument.
    