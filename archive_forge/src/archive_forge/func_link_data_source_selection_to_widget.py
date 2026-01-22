from __future__ import absolute_import
import ipywidgets as widgets
from bokeh.models import CustomJS
from traitlets import Unicode
import ipyvolume
def link_data_source_selection_to_widget(data_source, widget, trait_name):
    _ensure_widget_manager_hack()
    callback = CustomJS(args=dict(data=data_source), code='\n\n    var indices = data.selected["1d"].indices\n    var widget_id = \'%s\'\n    if(jupyter_widget_manager) {\n        // MYSTERY: if we use require, we end up at bokeh\'s require, which cannot find it, using requirejs it seems to work\n        requirejs(["@jupyter-widgets/base"], function(widgets) {\n            var widget_promise = widgets.unpack_models(\'IPY_MODEL_\' +widget_id, jupyter_widget_manager)\n            widget_promise.then(function(widget) {\n                     widget.set(%r, indices)\n                     widget.save_changes()\n            })\n        })\n    } else {\n        console.error("no widget manager")\n    }\n\n    ' % (widget.model_id, trait_name))
    data_source.selected.js_on_change('indices', callback)