from pyparsing import *
import random
import string
def breakItem(self):
    if not self.isBroken:
        print('<Crash!>')
        self.desc = 'broken ' + self.desc
        self.isBroken = True