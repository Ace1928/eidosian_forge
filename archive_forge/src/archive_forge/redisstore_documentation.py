import os
from json import loads, dumps
from kivy.compat import iteritems
from kivy.properties import StringProperty
from kivy.storage import AbstractStore
Store implementation using a Redis database.
    See the :mod:`kivy.storage` module documentation for more information.
    