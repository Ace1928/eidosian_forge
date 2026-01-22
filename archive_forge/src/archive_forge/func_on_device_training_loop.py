import contextlib
import threading
def on_device_training_loop(func):
    setattr(func, 'step_marker_location', 'STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP')
    return func