import ctypes
from ctypes import *
import pyglet.lib
Wrapper for pulse

Generated with:
tools/genwrappers.py pulseaudio

Do not modify this file.

!!! IMPORTANT !!!

Despite the warning up there, this file has been manually modified:
- struct_timeval, stemming from <sys/time.h>, is incorrectly parsed by
  tools/genwrappers.py and was manually edited and shifted to the top.
- All `pa_proplist_*` function definitions and the definition of an
  associated enum have been copypasted over from a different run of this
  script on a (likely) later version of PulseAudio's headers.
  - This includes modifiction of `__all__` at the very end of the file.
- All definitions of opaque structs (_opaque_struct dummy member) were
  duplicated. Those duplicates have been manually removed.
