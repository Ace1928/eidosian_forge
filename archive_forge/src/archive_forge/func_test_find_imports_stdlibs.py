import pathlib
from panel.io.mime_render import (
def test_find_imports_stdlibs():
    code = '\n    import os\n    import base64\n    import pathlib\n    import random\n    from datetime import datetime, timedelta\n    '
    assert find_requirements(code) == []