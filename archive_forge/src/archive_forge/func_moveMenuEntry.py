import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def moveMenuEntry(self, menuentry, oldparent, newparent, after=None, before=None):
    self.__deleteEntry(oldparent, menuentry, after, before)
    self.__addEntry(newparent, menuentry, after, before)
    self.menu.sort()
    return menuentry