import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def createMenuEntry(self, parent, name, command=None, genericname=None, comment=None, icon=None, terminal=None, after=None, before=None):
    menuentry = MenuEntry(self.__getFileName(name, '.desktop'))
    menuentry = self.editMenuEntry(menuentry, name, genericname, comment, command, icon, terminal)
    self.__addEntry(parent, menuentry, after, before)
    self.menu.sort()
    return menuentry