import contextlib
import sys
import tempfile
from glob import glob
import os
from shutil import rmtree
import textwrap
import typing
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface as ri
import rpy2.rinterface_lib.openrlib
import rpy2.robjects as ro
import rpy2.robjects.packages as rpacks
from rpy2.robjects.lib import grdevices
from rpy2.robjects.conversion import (Converter,
import warnings
import IPython.display  # type: ignore
from IPython.core import displaypub  # type: ignore
from IPython.core.magic import (Magics,   # type: ignore
from IPython.core.magic_arguments import (argument,  # type: ignore

        Execute code in R, optionally returning results to the Python runtime.

        In line mode, this will evaluate an expression and convert the returned
        value to a Python object.  The return value is determined by rpy2's
        behaviour of returning the result of evaluating the final expression.

        Multiple R expressions can be executed by joining them with
        semicolons::

            In [9]: %R X=c(1,4,5,7); sd(X); mean(X)
            Out[9]: array([ 4.25])

        In cell mode, this will run a block of R code. The resulting value
        is printed if it would be printed when evaluating the same code
        within a standard R REPL.

        Nothing is returned to python by default in cell mode::

            In [10]: %%R
               ....: Y = c(2,4,3,9)
               ....: summary(lm(Y~X))

            Call:
            lm(formula = Y ~ X)

            Residuals:
                1     2     3     4
             0.88 -0.24 -2.28  1.64

            Coefficients:
                        Estimate Std. Error t value Pr(>|t|)
            (Intercept)   0.0800     2.3000   0.035    0.975
            X             1.0400     0.4822   2.157    0.164

            Residual standard error: 2.088 on 2 degrees of freedom
            Multiple R-squared: 0.6993,Adjusted R-squared: 0.549
            F-statistic: 4.651 on 1 and 2 DF,  p-value: 0.1638

        In the notebook, plots are published as the output of the cell::

            %R plot(X, Y)

        will create a scatter plot of X bs Y.

        If cell is not None and line has some R code, it is prepended to
        the R code in cell.

        Objects can be passed back and forth between rpy2 and python via the
        -i -o flags in line::

            In [14]: Z = np.array([1,4,5,10])

            In [15]: %R -i Z mean(Z)
            Out[15]: array([ 5.])


            In [16]: %R -o W W=Z*mean(Z)
            Out[16]: array([  5.,  20.,  25.,  50.])

            In [17]: W
            Out[17]: array([  5.,  20.,  25.,  50.])

        The return value is determined by these rules:

        * If the cell is not None (i.e., has contents), the magic returns None.

        * If the final line results in a NULL value when evaluated
          by rpy2, then None is returned.

        * No attempt is made to convert the final value to a structured array.
          Use %Rget to push a structured array.

        * If the -n flag is present, there is no return value.

        * A trailing ';' will also result in no return value as the last
          value in the line is an empty string.
        