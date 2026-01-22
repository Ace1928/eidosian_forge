import threading
import typing
import warnings
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def getprime_func(nbits: int) -> int:
    return parallel.getprime(nbits, poolsize=poolsize)