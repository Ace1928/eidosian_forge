from math import pi, sin
from struct import pack
from PySide2.QtCore import QByteArray, QIODevice, Qt, QTimer, qWarning
from PySide2.QtMultimedia import (QAudio, QAudioDeviceInfo, QAudioFormat,
from PySide2.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel,
def toggleSuspendResume(self):
    if self.m_audioOutput.state() == QAudio.SuspendedState:
        qWarning('status: Suspended, resume()')
        self.m_audioOutput.resume()
        self.m_suspendResumeButton.setText(self.SUSPEND_LABEL)
    elif self.m_audioOutput.state() == QAudio.ActiveState:
        qWarning('status: Active, suspend()')
        self.m_audioOutput.suspend()
        self.m_suspendResumeButton.setText(self.RESUME_LABEL)
    elif self.m_audioOutput.state() == QAudio.StoppedState:
        qWarning('status: Stopped, resume()')
        self.m_audioOutput.resume()
        self.m_suspendResumeButton.setText(self.SUSPEND_LABEL)
    elif self.m_audioOutput.state() == QAudio.IdleState:
        qWarning('status: IdleState')