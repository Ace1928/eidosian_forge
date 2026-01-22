from appearing. If you want to prevent the settings instance from appearing
from the same thread and the other coroutines are only executed when Kivy
import os
from inspect import getfile
from os.path import dirname, join, exists, sep, expanduser, isfile
from kivy.config import ConfigParser
from kivy.base import runTouchApp, async_runTouchApp, stopTouchApp
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty
from kivy.setupconfig import USE_SDL2
def get_application_config(self, defaultpath='%(appdir)s/%(appname)s.ini'):
    """
        Return the filename of your application configuration. Depending
        on the platform, the application file will be stored in
        different locations:

            - on iOS: <appdir>/Documents/.<appname>.ini
            - on Android: <user_data_dir>/.<appname>.ini
            - otherwise: <appdir>/<appname>.ini

        When you are distributing your application on Desktops, please
        note that if the application is meant to be installed
        system-wide, the user might not have write-access to the
        application directory. If you want to store user settings, you
        should overload this method and change the default behavior to
        save the configuration file in the user directory. ::

            class TestApp(App):
                def get_application_config(self):
                    return super(TestApp, self).get_application_config(
                        '~/.%(appname)s.ini')

        Some notes:

        - The tilda '~' will be expanded to the user directory.
        - %(appdir)s will be replaced with the application :attr:`directory`
        - %(appname)s will be replaced with the application :attr:`name`

        .. versionadded:: 1.0.7

        .. versionchanged:: 1.4.0
            Customized the defaultpath for iOS and Android platforms. Added a
            defaultpath parameter for desktop OS's (not applicable to iOS
            and Android.)

        .. versionchanged:: 1.11.0
            Changed the Android version to make use of the
            :attr:`~App.user_data_dir` and added a missing dot to the iOS
            config file name.
        """
    if platform == 'android':
        return join(self.user_data_dir, '.{0}.ini'.format(self.name))
    elif platform == 'ios':
        defaultpath = '~/Documents/.%(appname)s.ini'
    elif platform == 'win':
        defaultpath = defaultpath.replace('/', sep)
    return expanduser(defaultpath) % {'appname': self.name, 'appdir': self.directory}