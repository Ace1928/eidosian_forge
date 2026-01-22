import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
def plink_menus(self):
    """Menus for the SnapPyLinkEditor."""
    self.menubar = menubar = Tk_.Menu(self.window)
    if sys.platform == 'darwin':
        Python_menu = Tk_.Menu(menubar, name='apple')
        Python_menu.add_command(label='About SnapPy...', command=self.main_window.about_window)
        menubar.add_cascade(label='SnapPy', menu=Python_menu)
    File_menu = Tk_.Menu(menubar, name='file')
    add_menu(self.window, File_menu, 'Open...', self.load)
    add_menu(self.window, File_menu, 'Save as...', self.save)
    self.build_save_image_menu(menubar, File_menu)
    File_menu.add_separator()
    if self.callback:
        add_menu(self.window, File_menu, 'Close', self.done)
    else:
        add_menu(self.window, File_menu, 'Exit', self.done)
    menubar.add_cascade(label='File', menu=File_menu)
    Edit_menu = Tk_.Menu(menubar, name='snappyedit')
    add_menu(self.window, Edit_menu, 'Cut', None, state='disabled')
    add_menu(self.window, Edit_menu, 'Copy', None, state='disabled')
    add_menu(self.window, Edit_menu, 'Paste', None, state='disabled')
    add_menu(self.window, Edit_menu, 'Delete', None, state='disabled')
    menubar.add_cascade(label='Edit ', menu=Edit_menu)
    self._add_info_menu()
    self._add_tools_menu()
    self._add_style_menu()
    menubar.add_cascade(label='Window', menu=WindowMenu(menubar))
    Help_menu = Tk_.Menu(menubar, name='help')
    menubar.add_cascade(label='Help', menu=HelpMenu(menubar))
    Help_menu.add_command(label='PLink Help ...', command=self.howto)
    self.window.config(menu=menubar)