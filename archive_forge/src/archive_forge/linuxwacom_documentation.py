import os
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect

Native support of Wacom tablet from linuxwacom driver
=====================================================

To configure LinuxWacom, add this to your configuration::

    [input]
    pen = linuxwacom,/dev/input/event2,mode=pen
    finger = linuxwacom,/dev/input/event3,mode=touch

.. note::
    You must have read access to the input event.

You can use a custom range for the X, Y and pressure values.
On some drivers, the range reported is invalid.
To fix that, you can add these options to the argument line:

* invert_x : 1 to invert X axis
* invert_y : 1 to invert Y axis
* min_position_x : X minimum
* max_position_x : X maximum
* min_position_y : Y minimum
* max_position_y : Y maximum
* min_pressure : pressure minimum
* max_pressure : pressure maximum
