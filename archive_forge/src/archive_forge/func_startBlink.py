from ..Qt import QtCore, QtWidgets
def startBlink(self, color, message=None, tip='', limitedTime=True):
    self.setFixedHeight(self.height())
    if message is not None:
        self.setText(message, temporary=True)
    self.setToolTip(tip, temporary=True)
    self.count = 0
    self.indStyle = 'QPushButton {background-color: %s}' % color
    self.limitedTime = limitedTime
    self.borderOn()
    if limitedTime:
        QtCore.QTimer.singleShot(2000, self.setText)
        QtCore.QTimer.singleShot(10000, self.setToolTip)