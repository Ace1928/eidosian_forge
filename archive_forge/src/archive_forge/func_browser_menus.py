import sys
import os
import webbrowser
from urllib.request import pathname2url
from .gui import *
from . import __file__ as snappy_dir
from .infowindow import about_snappy, InfoWindow
from .version import version
import shutil
def browser_menus(self):
    """
    Menus for the browser window.  Used as Browser.build_menus.
    Creates a menubar attribute for the browser.
    """
    self.menubar = menubar = Tk_.Menu(self)
    if sys.platform == 'darwin' and self.main_window is not None:
        Python_menu = Tk_.Menu(menubar, name='apple')
        Python_menu.add_command(label='About SnapPy...', command=self.main_window.about_window)
        menubar.add_cascade(label='SnapPy', menu=Python_menu)
    File_menu = Tk_.Menu(menubar, name='file')
    add_menu(self, File_menu, 'Open...', None, 'disabled')
    add_menu(self, File_menu, 'Save as...', self.save)
    File_menu.add_separator()
    add_menu(self, File_menu, 'Close', self.close)
    menubar.add_cascade(label='File', menu=File_menu)
    menubar.add_cascade(label='Edit ', menu=EditMenu(menubar, self.edit_actions))
    if sys.platform == 'darwin':
        menubar.add_cascade(label='View', menu=Tk_.Menu(menubar, name='view'))
    menubar.add_cascade(label='Window', menu=WindowMenu(menubar))
    help_menu = HelpMenu(menubar)
    help_menu.extra_command(label=help_polyhedron_viewer_label, command=self.dirichlet_help)
    help_menu.extra_command(label=help_horoball_viewer_label, command=self.horoball_help)
    menubar.add_cascade(label='Help', menu=help_menu)