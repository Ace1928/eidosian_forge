import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest

        Pickling a C{lambda} function ought to raise a L{pickle.PicklingError}.
        