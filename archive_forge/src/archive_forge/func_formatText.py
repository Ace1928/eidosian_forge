import string
from twisted.logger import Logger
def formatText(self, text):
    return ColorText(text, self.pickColor(self.currentFG, 0), self.pickColor(self.currentBG, 1), self.display, self.bold, self.underline, self.flash, self.reverse)