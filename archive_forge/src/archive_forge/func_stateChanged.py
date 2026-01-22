import sys
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (qApp, QApplication, QComboBox, QFormLayout,
from PySide2.QtTextToSpeech import QTextToSpeech, QVoice
def stateChanged(self, state):
    if state == QTextToSpeech.State.Ready:
        self.sayButton.setEnabled(True)