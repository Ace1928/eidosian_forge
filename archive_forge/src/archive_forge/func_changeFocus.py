import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
def changeFocus(self):
    try:
        ContainerWidget.changeFocus(self)
    except YieldFocus:
        try:
            ContainerWidget.changeFocus(self)
        except YieldFocus:
            pass