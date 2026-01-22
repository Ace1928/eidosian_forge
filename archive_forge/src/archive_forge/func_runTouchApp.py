import sys
import os
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.context import register_context
def runTouchApp(widget=None, embedded=False):
    """Static main function that starts the application loop.
    You can access some magic via the following arguments:

    See :mod:`kivy.app` for example usage.

    :Parameters:
        `<empty>`
            To make dispatching work, you need at least one
            input listener. If not, application will leave.
            (MTWindow act as an input listener)

        `widget`
            If you pass only a widget, a MTWindow will be created
            and your widget will be added to the window as the root
            widget.

        `embedded`
            No event dispatching is done. This will be your job.

        `widget + embedded`
            No event dispatching is done. This will be your job but
            we try to get the window (must be created by you beforehand)
            and add the widget to it. Very useful for embedding Kivy
            in another toolkit. (like Qt, check kivy-designed)

    """
    _runTouchApp_prepare(widget=widget)
    if embedded:
        return
    try:
        EventLoop.mainloop()
    finally:
        stopTouchApp()