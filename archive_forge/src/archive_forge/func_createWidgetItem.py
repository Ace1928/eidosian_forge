import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def createWidgetItem(self, item_type, elem, getter, *getter_args):
    """ Create a specific type of widget item. """
    item = self.factory.createQObject(item_type, 'item', (), False)
    props = self.wprops
    text = props.getProperty(elem, 'text')
    status_tip = props.getProperty(elem, 'statusTip')
    tool_tip = props.getProperty(elem, 'toolTip')
    whats_this = props.getProperty(elem, 'whatsThis')
    if self.any_i18n(text, status_tip, tool_tip, whats_this):
        self.factory.invoke('item', getter, getter_args)
    if text:
        item.setText(text)
    if status_tip:
        item.setStatusTip(status_tip)
    if tool_tip:
        item.setToolTip(tool_tip)
    if whats_this:
        item.setWhatsThis(whats_this)
    text_alignment = props.getProperty(elem, 'textAlignment')
    if text_alignment:
        item.setTextAlignment(text_alignment)
    font = props.getProperty(elem, 'font')
    if font:
        item.setFont(font)
    icon = props.getProperty(elem, 'icon')
    if icon:
        item.setIcon(icon)
    background = props.getProperty(elem, 'background')
    if background:
        item.setBackground(background)
    foreground = props.getProperty(elem, 'foreground')
    if foreground:
        item.setForeground(foreground)
    flags = props.getProperty(elem, 'flags')
    if flags:
        item.setFlags(flags)
    check_state = props.getProperty(elem, 'checkState')
    if check_state:
        item.setCheckState(check_state)
    return item