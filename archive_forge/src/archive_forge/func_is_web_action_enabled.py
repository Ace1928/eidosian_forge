import sys
from PySide2.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
from PySide2 import QtCore
def is_web_action_enabled(self, web_action):
    return self.page().action(web_action).isEnabled()