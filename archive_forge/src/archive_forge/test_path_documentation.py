import io
import pathlib
import unittest
import importlib_resources as resources
from . import data01
from . import util

        It is not an error if the file that was temporarily stashed on the
        file system is removed inside the `with` stanza.
        