from PySide2 import QtWidgets
def createIntroPage():
    page = QtWidgets.QWizardPage()
    page.setTitle('Introduction')
    label = QtWidgets.QLabel('This wizard will help you register your copy of Super Product Two.')
    label.setWordWrap(True)
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(label)
    page.setLayout(layout)
    return page