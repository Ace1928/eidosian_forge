from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def notified(self):
    qWarning('bytesFree = %d, elapsedUSecs = %d, processedUSecs = %d' % (self.m_audioOutput.bytesFree(), self.m_audioOutput.elapsedUSecs(), self.m_audioOutput.processedUSecs()))