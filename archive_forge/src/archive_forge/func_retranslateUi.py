from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
from PySide2.QtGui import (QColor, QFont, QIcon, QPixmap)
from PySide2.QtWidgets import *
def retranslateUi(self, Form):
    Form.setWindowTitle(QCoreApplication.translate('Form', u'Easing curves', None))
    self.groupBox_2.setTitle(QCoreApplication.translate('Form', u'Path type', None))
    self.lineRadio.setText(QCoreApplication.translate('Form', u'Line', None))
    self.circleRadio.setText(QCoreApplication.translate('Form', u'Circle', None))
    self.groupBox.setTitle(QCoreApplication.translate('Form', u'Properties', None))
    self.label.setText(QCoreApplication.translate('Form', u'Period', None))
    self.label_2.setText(QCoreApplication.translate('Form', u'Amplitude', None))
    self.label_3.setText(QCoreApplication.translate('Form', u'Overshoot', None))