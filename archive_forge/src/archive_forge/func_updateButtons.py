import sys
from PySide2.QtCore import SLOT, QStandardPaths, Qt
from PySide2.QtGui import QIcon, QKeySequence
from PySide2.QtWidgets import (QAction, qApp, QApplication, QDialog, QFileDialog,
from PySide2.QtMultimedia import QMediaPlayer, QMediaPlaylist
from PySide2.QtMultimediaWidgets import QVideoWidget
def updateButtons(self, state):
    mediaCount = self.playlist.mediaCount()
    self.playAction.setEnabled(mediaCount > 0 and state != QMediaPlayer.PlayingState)
    self.pauseAction.setEnabled(state == QMediaPlayer.PlayingState)
    self.stopAction.setEnabled(state != QMediaPlayer.StoppedState)
    self.previousAction.setEnabled(self.player.position() > 0)
    self.nextAction.setEnabled(mediaCount > 1)