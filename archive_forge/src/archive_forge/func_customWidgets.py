import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def customWidgets(self, elem):

    def header2module(header):
        """header2module(header) -> string

            Convert paths to C++ header files to according Python modules
            >>> header2module("foo/bar/baz.h")
            'foo.bar.baz'
            """
        if header.endswith('.h'):
            header = header[:-2]
        mpath = []
        for part in header.split('/'):
            if part not in ('', '.'):
                if part == '..':
                    raise SyntaxError("custom widget header file name may not contain '..'.")
                mpath.append(part)
        return '.'.join(mpath)
    for custom_widget in iter(elem):
        classname = custom_widget.findtext('class')
        self.factory.addCustomWidget(classname, custom_widget.findtext('extends') or 'QWidget', header2module(custom_widget.findtext('header')))