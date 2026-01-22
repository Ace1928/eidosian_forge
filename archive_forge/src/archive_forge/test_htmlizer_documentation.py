from io import BytesIO
from twisted.python.htmlizer import filter
from twisted.trial.unittest import TestCase

        If passed an input file containing a variable access, L{filter} writes
        a I{pre} tag containing a I{py-src-variable} span containing the
        variable.
        