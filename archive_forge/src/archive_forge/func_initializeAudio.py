from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def initializeAudio(self):
    self.m_pullTimer = QTimer(self)
    self.m_pullTimer.timeout.connect(self.pullTimerExpired)
    self.m_pullMode = True
    self.m_format = QAudioFormat()
    self.m_format.setSampleRate(self.DataSampleRateHz)
    self.m_format.setChannelCount(1)
    self.m_format.setSampleSize(16)
    self.m_format.setCodec('audio/pcm')
    self.m_format.setByteOrder(QAudioFormat.LittleEndian)
    self.m_format.setSampleType(QAudioFormat.SignedInt)
    info = QAudioDeviceInfo(QAudioDeviceInfo.defaultOutputDevice())
    if not info.isFormatSupported(self.m_format):
        qWarning('Default format not supported - trying to use nearest')
        self.m_format = info.nearestFormat(self.m_format)
    self.m_generator = Generator(self.m_format, self.DurationSeconds * 1000000, self.ToneSampleRateHz, self)
    self.createAudioOutput()