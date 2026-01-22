import sys
import unittest
import platform
import pygame
def is_dict_alive(self):
    try:
        return self.dict_ref() is not None
    except AttributeError:
        raise NoDictError('__array_interface__ is unread')