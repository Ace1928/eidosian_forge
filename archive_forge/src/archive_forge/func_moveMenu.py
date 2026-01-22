import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def moveMenu(self, menu, oldparent, newparent, after=None, before=None):
    self.__deleteEntry(oldparent, menu, after, before)
    self.__addEntry(newparent, menu, after, before)
    root_menu = self.__getXmlMenu(self.menu.Name)
    if oldparent.getPath(True) != newparent.getPath(True):
        self.__addXmlMove(root_menu, os.path.join(oldparent.getPath(True), menu.Name), os.path.join(newparent.getPath(True), menu.Name))
    self.menu.sort()
    return menu