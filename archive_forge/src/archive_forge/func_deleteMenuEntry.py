import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def deleteMenuEntry(self, menuentry):
    if self.getAction(menuentry) == 'delete':
        self.__deleteFile(menuentry.DesktopEntry.filename)
        for parent in menuentry.Parents:
            self.__deleteEntry(parent, menuentry)
        self.menu.sort()
    return menuentry