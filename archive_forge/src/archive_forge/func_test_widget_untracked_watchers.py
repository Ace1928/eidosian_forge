import param
import pytest
from panel.io import block_comm
from panel.layout import Row
from panel.links import CallbackGenerator
from panel.tests.util import check_layoutable_properties
from panel.util import param_watchers
from panel.widgets import (
from panel.widgets.tables import BaseTable
@pytest.mark.parametrize('widget', all_widgets)
def test_widget_untracked_watchers(widget, document, comm):
    try:
        widg = widget()
    except ImportError:
        pytest.skip('Dependent library could not be imported.')
    watchers = [w for pwatchers in param_watchers(widg).values() for awatchers in pwatchers.values() for w in awatchers]
    assert len([wfn for wfn in watchers if wfn not in widg._internal_callbacks and (not hasattr(wfn.fn, '_watcher_name'))]) == 0