from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def deviceChanged(self, index):
    self.m_pullTimer.stop()
    self.m_generator.stop()
    self.m_audioOutput.stop()
    self.m_device = self.m_deviceBox.itemData(index)
    self.createAudioOutput()