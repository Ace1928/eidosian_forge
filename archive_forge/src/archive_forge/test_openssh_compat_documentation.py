import os
from unittest import skipIf
from twisted.conch.ssh._kex import getDHGeneratorAndPrime
from twisted.conch.test import keydata
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase

        L{OpenSSHFactory.getPrimes} should return the available primes
        in the moduli directory.
        