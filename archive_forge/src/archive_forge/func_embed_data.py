import json
import re
from .widgets import Widget, DOMWidget, widget as widget_module
from .widgets.widget_link import Link
from .widgets.docutils import doc_subst
from ._version import __html_manager_version__
@doc_subst(_doc_snippets)
def embed_data(views, drop_defaults=True, state=None):
    """Gets data for embedding.

    Use this to get the raw data for embedding if you have special
    formatting needs.

    Parameters
    ----------
    {views_attribute}
    drop_defaults: boolean
        Whether to drop default values from the widget states.
    state: dict or None (default)
        The state to include. When set to None, the state of all widgets
        know to the widget manager is included. Otherwise it uses the
        passed state directly. This allows for end users to include a
        smaller state, under the responsibility that this state is
        sufficient to reconstruct the embedded views.

    Returns
    -------
    A dictionary with the following entries:
        manager_state: dict of the widget manager state data
        view_specs: a list of widget view specs
    """
    if views is None:
        views = [w for w in widget_module._instances.values() if isinstance(w, DOMWidget)]
    else:
        try:
            views[0]
        except (IndexError, TypeError):
            views = [views]
    if state is None:
        state = Widget.get_manager_state(drop_defaults=drop_defaults, widgets=None)['state']
    json_data = Widget.get_manager_state(widgets=[])
    json_data['state'] = state
    view_specs = [w.get_view_spec() for w in views]
    return dict(manager_state=json_data, view_specs=view_specs)