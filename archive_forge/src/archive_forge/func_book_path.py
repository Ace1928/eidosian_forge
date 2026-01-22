import abc
from taskflow import exceptions as exc
from taskflow.persistence import base
from taskflow.persistence import models
@property
def book_path(self):
    return self._book_path