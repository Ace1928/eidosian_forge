import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
@requires_graphics
@pytest.mark.parametrize('widget_verify_thread', ['0', '1'], indirect=True)
def test_change_graphics_second_thread(widget_verify_thread):
    from kivy.graphics import Color
    widget, verify_thread = widget_verify_thread
    with widget.canvas:
        color = Color()
    exception = None

    def callback():
        nonlocal exception
        try:
            if verify_thread == '1':
                with pytest.raises(TypeError):
                    color.rgb = (0.1, 0.2, 0.3)
            else:
                color.rgb = (0.1, 0.2, 0.3)
        except BaseException as e:
            exception = (e, sys.exc_info()[2])
            raise
    thread = Thread(target=callback)
    thread.start()
    thread.join()
    if exception is not None:
        raise exception[0].with_traceback(exception[1])