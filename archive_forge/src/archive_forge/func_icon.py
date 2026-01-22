import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
@icon.setter
def icon(self, icon):
    self._icon = icon
    self.set_value('app', 'icon', icon)