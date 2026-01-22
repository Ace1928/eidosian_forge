from traitlets.config.configurable import SingletonConfigurable
from traitlets import (
def pil_available():
    """Test if PIL/Pillow is available"""
    out = False
    try:
        from PIL import Image
        out = True
    except ImportError:
        pass
    return out