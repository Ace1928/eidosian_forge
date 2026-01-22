import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def buttonGroups(self, elem):
    for button_group in iter(elem):
        if button_group.tag == 'buttongroup':
            bg_name = button_group.attrib['name']
            bg = ButtonGroup()
            self.button_groups[bg_name] = bg
            prop = self.getProperty(button_group, 'exclusive')
            if prop is not None:
                if prop.findtext('bool') == 'false':
                    bg.exclusive = False