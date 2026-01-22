import re
import itertools
@staticmethod
def getErrorCorrectPolynomial(errorCorrectLength):
    a = QRPolynomial([1], 0)
    for i in range(errorCorrectLength):
        a = a.multiply(QRPolynomial([1, QRMath.gexp(i)], 0))
    return a