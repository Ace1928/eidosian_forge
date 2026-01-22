import json
import os
import sys
import uuid
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
import param
from bokeh.core.property.bases import Property
from bokeh.models import CustomJS
from param.parameterized import Watcher
from ..util import param_watchers
from .model import add_to_doc, diff
from .state import state
def embed_state(panel, model, doc, max_states=1000, max_opts=3, json=False, json_prefix='', save_path='./', load_path=None, progress=True, states={}):
    """
    Embeds the state of the application on a State model which allows
    exporting a static version of an app. This works by finding all
    widgets with a predefined set of options and evaluating the cross
    product of the widget values and recording the resulting events to
    be replayed when exported. The state is recorded on a State model
    which is attached as an additional root on the Document.

    Arguments
    ---------
    panel: panel.reactive.Reactive
      The Reactive component being exported
    model: bokeh.model.Model
      The bokeh model being exported
    doc: bokeh.document.Document
      The bokeh Document being exported
    max_states: int (default=1000)
      The maximum number of states to export
    max_opts: int (default=3)
      The max number of ticks sampled in a continuous widget like a slider
    json: boolean (default=True)
      Whether to export the data to json files
    json_prefix: str (default='')
      Prefix for JSON filename
    save_path: str (default='./')
      The path to save json files to
    load_path: str (default=None)
      The path or URL the json files will be loaded from.
    progress: boolean (default=True)
      Whether to report progress
    states: dict (default={})
      A dictionary specifying the widget values to embed for each widget
    """
    from tqdm import tqdm
    from ..config import config
    from ..layout import Panel
    from ..links import Link
    from ..models.state import State
    from ..pane import PaneBase
    from ..widgets import DiscreteSlider, Widget
    ref = model.ref['id']
    if isinstance(panel, PaneBase) and ref in panel.layout._models:
        panel = panel.layout
    if not isinstance(panel, Panel):
        add_to_doc(model, doc)
        return
    _, _, _, comm = state._views[ref]
    model.tags.append('embedded')

    def is_embeddable(object):
        if not isinstance(object, Widget) or object.disabled:
            return False
        if isinstance(object, DiscreteSlider):
            return ref in object._composite[1]._models
        return ref in object._models
    widgets = [w for w in panel.select(is_embeddable) if w not in Link.registry]
    state_model = State()
    widget_data, merged, ignore = ([], {}, [])
    for widget in widgets:
        if 'composite' in widget.tags:
            continue
        if widget._param_pane is not None:
            link = param_to_jslink(model, widget)
            if link is not None:
                pobj = widget._param_pane.object
                if isinstance(pobj, Widget):
                    if not any((w not in pobj._internal_callbacks and w not in widget._param_pane._internal_callbacks for w in get_watchers(pobj))):
                        ignore.append(pobj)
                continue
        if widget._links:
            jslinks = links_to_jslinks(model, widget)
            if jslinks:
                continue
        if not widget._supports_embed or all((w in widget._internal_callbacks for w in get_watchers(widget))):
            continue
        w, w_model, vals, getter, on_change, js_getter = widget._get_embed_state(model, states.get(widget), max_opts)
        w_type = w._widget_type
        if isinstance(w, DiscreteSlider):
            w_model = w._composite[1]._models[ref][0].select_one({'type': w_type})
        else:
            w_model = w._models[ref][0]
            if not isinstance(w_model, w_type):
                w_model = w_model.select_one({'type': w_type})
        if widget.name and widget.name in merged:
            merged[widget.name][0].append(w)
            merged[widget.name][1].append(w_model)
            continue
        js_callback = CustomJS(code=STATE_JS.format(id=state_model.ref['id'], js_getter=js_getter))
        widget_data.append(([w], [w_model], vals, getter, js_callback, on_change))
        merged[widget.name] = widget_data[-1]
    values = []
    for ws, w_models, vals, getter, js_callback, on_change in widget_data:
        if ws[0] in ignore:
            continue
        for w_model in w_models:
            w_model.js_on_change(on_change, js_callback)
        wm = w_models[0]
        for wmo in w_models[1:]:
            attr = ws[0]._rename.get('value', 'value')
            wm.js_link(attr, wmo, attr)
            wmo.js_link(attr, wm, attr)
        values.append((ws, w_models, vals, getter))
    add_to_doc(model, doc, True)
    doc.callbacks._held_events = []
    if not widget_data:
        return
    restore = [ws[0].value for ws, _, _, _ in values]
    init_vals = [g(ms[0]) for _, ms, _, g in values]
    cross_product = list(product(*[vals[::-1] for _, _, vals, _ in values]))
    if len(cross_product) > max_states:
        if config._doc_build:
            return
        param.main.param.warning('The cross product of different application states is very large to explore (N=%d), consider reducing the number of options on the widgets or increase the max_states specified in the function to remove this warning' % len(cross_product))
    nested_dict = lambda: defaultdict(nested_dict)
    state_dict = nested_dict()
    changes = False
    for key in tqdm(cross_product, leave=False, file=sys.stdout) if progress else cross_product:
        sub_dict = state_dict
        skip = False
        for i, k in enumerate(key):
            ws, m, _, g = values[i]
            try:
                with always_changed(config.safe_embed):
                    for w in ws:
                        w.value = k
            except Exception:
                skip = True
                break
            sub_dict = sub_dict[g(m[0])]
        if skip:
            doc.callbacks._held_events = []
            continue
        models = [m for v in values for m in v[1]]
        doc.callbacks._held_events = [e for e in doc.callbacks._held_events if e.model not in models]
        events = record_events(doc)
        changes |= events['content'] != '{}'
        if events:
            sub_dict.update(events)
    if not changes:
        return
    for (ws, _, _, _), v in zip(values, restore):
        try:
            for w in ws:
                w.param.update(value=v)
        except Exception:
            pass
    if json:
        random_dir = '_'.join([json_prefix, uuid.uuid4().hex])
        save_path = os.path.join(save_path, random_dir)
        if load_path is not None:
            load_path = os.path.join(load_path, random_dir)
        state_dict = save_dict(state_dict, max_depth=len(values) - 1, save_path=save_path, load_path=load_path)
    state_model.update(json=json, state=state_dict, values=init_vals, widgets={m[0].ref['id']: i for i, (_, m, _, _) in enumerate(values)})
    doc.add_root(state_model)
    return state_model