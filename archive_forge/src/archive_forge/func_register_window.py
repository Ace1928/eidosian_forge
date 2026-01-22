from kivy.config import Config
from kivy.logger import Logger
import kivy
import importlib
import os
import sys
def register_window(self, win):
    """Add the window to the window list"""
    if win not in self.wins:
        self.wins.append(win)
    self.update()