import io
from IPython.core import page
def test_detect_screen_size():
    """Simple smoketest for page._detect_screen_size."""
    try:
        page._detect_screen_size(True, 25)
    except (TypeError, io.UnsupportedOperation):
        pass