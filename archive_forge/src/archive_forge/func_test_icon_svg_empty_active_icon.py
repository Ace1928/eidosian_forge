import pytest
from panel.widgets.icon import ToggleIcon
def test_icon_svg_empty_active_icon(self):
    with pytest.raises(ValueError, match='The active_icon parameter must not '):
        ToggleIcon(icon='<svg></svg>')