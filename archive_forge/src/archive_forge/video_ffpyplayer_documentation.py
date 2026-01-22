from threading import Thread
from queue import Queue, Empty, Full
from kivy.clock import Clock, mainthread
from kivy.logger import Logger
from kivy.core.video import VideoBase
from kivy.graphics import Rectangle, BindTexture
from kivy.graphics.texture import Texture
from kivy.graphics.fbo import Fbo
from kivy.weakmethod import WeakMethod
import time

FFmpeg based video abstraction
==============================

To use, you need to install ffpyplayer and have a compiled ffmpeg shared
library.

    https://github.com/matham/ffpyplayer

The docs there describe how to set this up. But briefly, first you need to
compile ffmpeg using the shared flags while disabling the static flags (you'll
probably have to set the fPIC flag, e.g. CFLAGS=-fPIC). Here are some
instructions: https://trac.ffmpeg.org/wiki/CompilationGuide. For Windows, you
can download compiled GPL binaries from http://ffmpeg.zeranoe.com/builds/.
Similarly, you should download SDL2.

Now, you should have ffmpeg and sdl directories. In each, you should have an
'include', 'bin' and 'lib' directory, where e.g. for Windows, 'lib' contains
the .dll.a files, while 'bin' contains the actual dlls. The 'include' directory
holds the headers. The 'bin' directory is only needed if the shared libraries
are not already in the path. In the environment, define FFMPEG_ROOT and
SDL_ROOT, each pointing to the ffmpeg and SDL directories respectively. (If
you're using SDL2, the 'include' directory will contain an 'SDL2' directory,
which then holds the headers).

Once defined, download the ffpyplayer git repo and run

    python setup.py build_ext --inplace

Finally, before running you need to ensure that ffpyplayer is in python's path.

..Note::

    When kivy exits by closing the window while the video is playing,
    it appears that the __del__method of VideoFFPy
    is not called. Because of this, the VideoFFPy object is not
    properly deleted when kivy exits. The consequence is that because
    MediaPlayer creates internal threads which do not have their daemon
    flag set, when the main threads exists, it'll hang and wait for the other
    MediaPlayer threads to exit. But since __del__ is not called to delete the
    MediaPlayer object, those threads will remain alive, hanging kivy. What
    this means is that you have to be sure to delete the MediaPlayer object
    before kivy exits by setting it to None.
