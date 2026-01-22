import os
from xdg.Menu import Menu, MenuEntry, Layout, Separator, XMLMenuBuilder
from xdg.BaseDirectory import xdg_config_dirs, xdg_data_dirs
from xdg.Exceptions import ParsingError 
from xdg.Config import setRootMode
def unhideMenu(self, menu):
    self.editMenu(menu, nodisplay=False, hidden=False)
    xml_menu = self.__getXmlMenu(menu.getPath(True, True), False)
    deleted = xml_menu.findall('Deleted')
    not_deleted = xml_menu.findall('NotDeleted')
    for node in deleted + not_deleted:
        xml_menu.remove(node)