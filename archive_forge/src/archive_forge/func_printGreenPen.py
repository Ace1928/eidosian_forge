from fontTools.pens.basePen import BasePen
from functools import partial
from itertools import count
import sympy as sp
import sys
def printGreenPen(penName, funcs, file=sys.stdout, docstring=None):
    if docstring is not None:
        print('"""%s"""' % docstring)
    print('from fontTools.pens.basePen import BasePen, OpenContourError\ntry:\n\timport cython\n\n\tCOMPILED = cython.compiled\nexcept (AttributeError, ImportError):\n\t# if cython not installed, use mock module with no-op decorators and types\n\tfrom fontTools.misc import cython\n\n\tCOMPILED = False\n\n\n__all__ = ["%s"]\n\nclass %s(BasePen):\n\n\tdef __init__(self, glyphset=None):\n\t\tBasePen.__init__(self, glyphset)\n' % (penName, penName), file=file)
    for name, f in funcs:
        print('\t\tself.%s = 0' % name, file=file)
    print('\n\tdef _moveTo(self, p0):\n\t\tself.__startPoint = p0\n\n\tdef _closePath(self):\n\t\tp0 = self._getCurrentPoint()\n\t\tif p0 != self.__startPoint:\n\t\t\tself._lineTo(self.__startPoint)\n\n\tdef _endPath(self):\n\t\tp0 = self._getCurrentPoint()\n\t\tif p0 != self.__startPoint:\n\t\t\t# Green theorem is not defined on open contours.\n\t\t\traise OpenContourError(\n\t\t\t\t\t\t\t"Green theorem is not defined on open contours."\n\t\t\t)\n', end='', file=file)
    for n in (1, 2, 3):
        subs = {P[i][j]: [X, Y][j][i] for i in range(n + 1) for j in range(2)}
        greens = [green(f, BezierCurve[n]) for name, f in funcs]
        greens = [sp.gcd_terms(f.collect(sum(P, ()))) for f in greens]
        greens = [f.subs(subs) for f in greens]
        defs, exprs = sp.cse(greens, optimizations='basic', symbols=(sp.Symbol('r%d' % i) for i in count()))
        print()
        for name, value in defs:
            print('\t@cython.locals(%s=cython.double)' % name, file=file)
        if n == 1:
            print('\t@cython.locals(x0=cython.double, y0=cython.double)\n\t@cython.locals(x1=cython.double, y1=cython.double)\n\tdef _lineTo(self, p1):\n\t\tx0,y0 = self._getCurrentPoint()\n\t\tx1,y1 = p1\n', file=file)
        elif n == 2:
            print('\t@cython.locals(x0=cython.double, y0=cython.double)\n\t@cython.locals(x1=cython.double, y1=cython.double)\n\t@cython.locals(x2=cython.double, y2=cython.double)\n\tdef _qCurveToOne(self, p1, p2):\n\t\tx0,y0 = self._getCurrentPoint()\n\t\tx1,y1 = p1\n\t\tx2,y2 = p2\n', file=file)
        elif n == 3:
            print('\t@cython.locals(x0=cython.double, y0=cython.double)\n\t@cython.locals(x1=cython.double, y1=cython.double)\n\t@cython.locals(x2=cython.double, y2=cython.double)\n\t@cython.locals(x3=cython.double, y3=cython.double)\n\tdef _curveToOne(self, p1, p2, p3):\n\t\tx0,y0 = self._getCurrentPoint()\n\t\tx1,y1 = p1\n\t\tx2,y2 = p2\n\t\tx3,y3 = p3\n', file=file)
        for name, value in defs:
            print('\t\t%s = %s' % (name, value), file=file)
        print(file=file)
        for name, value in zip([f[0] for f in funcs], exprs):
            print('\t\tself.%s += %s' % (name, value), file=file)
    print("\nif __name__ == '__main__':\n\tfrom fontTools.misc.symfont import x, y, printGreenPen\n\tprintGreenPen('%s', [" % penName, file=file)
    for name, f in funcs:
        print("\t\t      ('%s', %s)," % (name, str(f)), file=file)
    print('\t\t     ])', file=file)