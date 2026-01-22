from kivy.config import Config
from kivy.logger import Logger
import kivy
import importlib
import os
import sys
def unregister_window(self, win):
    """Remove the window from the window list"""
    if win in self.wins:
        self.wins.remove(win)
    self.update()