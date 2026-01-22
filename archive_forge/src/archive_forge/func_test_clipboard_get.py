from IPython.core.error import TryNext
from IPython.lib.clipboard import ClipboardEmpty
from IPython.testing.decorators import skip_if_no_x11
@skip_if_no_x11
def test_clipboard_get():
    try:
        a = get_ipython().hooks.clipboard_get()
    except ClipboardEmpty:
        pass
    except TryNext:
        pass
    else:
        assert isinstance(a, str)