from sys import argv
from kivy.lang import Builder
from kivy.app import App
from kivy.core.window import Window
from kivy.clock import Clock, mainthread
from kivy.uix.label import Label
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from os.path import dirname, basename, join

KViewer
=======

KViewer, for KV-Viewer, is a simple tool allowing you to dynamically display
a KV file, taking its changes into account (thanks to watchdog). The
idea is to facilitate design using the KV language. It's somewhat related to
the KivyCatalog demo, except it uses an on-disc file, allowing the user to use
any editor.

You can use the script as follows::

    python kviewer.py ./test.kv

This will display the test.kv and automatically update the display when the
file changes.

.. note: This scripts uses watchdog to listen for file changes. To install
   watchdog::

   pip install watchdog

