import errno
from os.path import exists, abspath, dirname
from kivy.compat import iteritems
from kivy.storage import AbstractStore
from json import loads, dump
Store implementation using a json file for storing the key-value pairs.
    See the :mod:`kivy.storage` module documentation for more information.
    