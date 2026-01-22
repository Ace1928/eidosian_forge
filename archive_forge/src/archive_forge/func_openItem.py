from pyparsing import *
import random
import string
def openItem(self, player):
    if not self.isOpened:
        self.isOpened = not self.isOpened
        if self.contents is not None:
            for item in self.contents:
                player.room.addItem(item)
            self.contents = []
        self.desc = 'open ' + self.desc