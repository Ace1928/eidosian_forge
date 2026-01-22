import re
from html.parser import HTMLParser
from html import entities
import pyglet
from pyglet.text.formats import structured
def prepare_for_data(self):
    if self.need_block_begin:
        self.add_text('\n')
        self.block_begin = True
        self.need_block_begin = False