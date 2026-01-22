from pyparsing import *
import random
import string
def closeItem(self, player):
    if self.isOpened:
        self.isOpened = not self.isOpened
        if self.desc.startswith('open '):
            self.desc = self.desc[5:]