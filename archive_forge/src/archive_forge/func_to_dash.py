import base64
import copy
import pickle
import uuid
from collections import namedtuple
from dash.exceptions import PreventUpdate
import holoviews as hv
from holoviews.core.decollate import (
from holoviews.plotting.plotly import DynamicMap, PlotlyRenderer
from holoviews.plotting.plotly.callbacks import (
from holoviews.plotting.plotly.util import clean_internal_figure_properties
from holoviews.streams import Derived, History
import plotly.graph_objects as go
from dash import callback_context
from dash.dependencies import Input, Output, State
def to_dash(app, hvobjs, reset_button=False, graph_class=dcc.Graph, button_class=html.Button, responsive='width', use_ranges=True):
    """
    Build Dash components and callbacks from a collection of HoloViews objects

    Args:
        app: dash.Dash application instance
        hvobjs: List of HoloViews objects to build Dash components from
        reset_button: If True, construct a Button component that, which clicked, will
            reset the interactive stream values associated with the provided HoloViews
            objects to their initial values. Defaults to False.
        graph_class: Class to use when creating Graph components, one of dcc.Graph
            (default) or ddk.Graph.
        button_class: Class to use when creating reset button component.
            E.g. html.Button (default) or dbc.Button
        responsive: If True graphs will fill their containers width and height
            responsively. If False, graphs will have a fixed size matching their
            HoloViews size. If "width" (default), the width is responsive but
            height matches the HoloViews size. If "height", the height is responsive
            but the width matches the HoloViews size.
        use_ranges: If True, initialize graphs with the dimension ranges specified
            in the HoloViews objects. If False, allow Dash to perform its own
            auto-range calculations.
    Returns:
        DashComponents named tuple with properties:
            - graphs: List of graph components (with type matching the input
                graph_class argument) with order corresponding to the order
                of the input hvobjs list.
            - resets: List of reset buttons that can be used to reset figure state.
                List has length 1 if reset_button=True and is empty if
                reset_button=False.
            - kdims: Dict from kdim names to Dash Components that can be used to
                set the corresponding kdim value.
            - store: dcc.Store the must be included in the app layout
            - children: Single list of all components above. The order is graphs,
                kdims, resets, and then the store.
    """
    num_figs = len(hvobjs)
    reset_components = []
    graph_components = []
    kdim_components = {}
    outputs = []
    inputs = []
    states = []
    plots = []
    graph_ids = []
    initial_fig_dicts = []
    all_kdims = {}
    kdims_per_fig = []
    uid_to_stream_ids = {}
    fig_to_fn_stream = {}
    fig_to_fn_stream_ids = {}
    plotly_stream_types = [RangeXYCallback, RangeXCallback, RangeYCallback, Selection1DCallback, BoundsXYCallback, BoundsXCallback, BoundsYCallback]
    layout_ranges = []
    for i, hvobj in enumerate(hvobjs):
        fn_spec = to_function_spec(hvobj)
        fig_to_fn_stream[i] = fn_spec
        kdims_per_fig.append(list(fn_spec.kdims))
        all_kdims.update(fn_spec.kdims)
        plot = PlotlyRenderer.get_plot(hvobj)
        plots.append(plot)
        layout_ranges.append(get_layout_ranges(plot))
        fig = plot_to_figure(plot, reset_nclicks=0, layout_ranges=layout_ranges[-1], responsive=responsive, use_ranges=use_ranges).to_dict()
        initial_fig_dicts.append(fig)
        graph_id = 'graph-' + str(uuid.uuid4())
        graph_ids.append(graph_id)
        graph = graph_class(id=graph_id, figure=fig, config={'scrollZoom': True})
        graph_components.append(graph)
        plotly_streams = {}
        for plotly_stream_type in plotly_stream_types:
            for t in fig['data']:
                if t.get('uid', None) in plotly_stream_type.instances:
                    plotly_streams.setdefault(plotly_stream_type, {})[t['uid']] = plotly_stream_type.instances[t['uid']]
        for plotly_stream_type, streams_for_type in plotly_streams.items():
            for uid, cb in streams_for_type.items():
                uid_to_stream_ids.setdefault(plotly_stream_type, {}).setdefault(uid, []).extend([id(stream) for stream in cb.streams])
        outputs.append(Output(component_id=graph_id, component_property='figure'))
        inputs.extend([Input(component_id=graph_id, component_property='selectedData'), Input(component_id=graph_id, component_property='relayoutData')])
    store_data = {'streams': {}}
    store_id = 'store-' + str(uuid.uuid4())
    states.append(State(store_id, 'data'))
    for fn_spec in fig_to_fn_stream.values():
        populate_store_with_stream_contents(store_data, fn_spec.streams)
    stream_callbacks = {}
    for fn_spec in fig_to_fn_stream.values():
        populate_stream_callback_graph(stream_callbacks, fn_spec.streams)
    for i, fn_spec in fig_to_fn_stream.items():
        fig_to_fn_stream_ids[i] = (fn_spec.fn, [id(stream) for stream in fn_spec.streams])
    store = dcc.Store(id=store_id, data=encode_store_data(store_data))
    outputs.append(Output(store_id, 'data'))
    initial_stream_contents = copy.deepcopy(store_data['streams'])
    kdim_uuids = []
    for kdim_name, (kdim_label, kdim_range) in all_kdims.items():
        slider_uuid = str(uuid.uuid4())
        slider_id = kdim_name + '-' + slider_uuid
        slider_label_id = kdim_name + '-label-' + slider_uuid
        kdim_uuids.append(slider_uuid)
        html_label = html.Label(id=slider_label_id, children=kdim_label)
        if isinstance(kdim_range, list):
            slider = html.Div(children=[html_label, dcc.Slider(id=slider_id, min=kdim_range[0], max=kdim_range[-1], step=None, marks={m: '' for m in kdim_range}, value=kdim_range[0])])
        else:
            slider = html.Div(children=[html_label, dcc.Slider(id=slider_id, min=kdim_range[0], max=kdim_range[-1], step=(kdim_range[-1] - kdim_range[0]) / 11.0, value=kdim_range[0])])
        kdim_components[kdim_name] = slider
        inputs.append(Input(component_id=slider_id, component_property='value'))
    if reset_button:
        reset_id = 'reset-' + str(uuid.uuid4())
        reset_button = button_class(id=reset_id, children='Reset')
        inputs.append(Input(component_id=reset_id, component_property='n_clicks'))
        reset_components.append(reset_button)

    @app.callback(outputs, inputs, states)
    def update_figure(*args):
        triggered_prop_ids = {entry['prop_id'] for entry in callback_context.triggered}
        selected_dicts = [args[j] or {} for j in range(0, num_figs * 2, 2)]
        relayout_dicts = [args[j] or {} for j in range(1, num_figs * 2, 2)]
        any_change = False
        store_data = decode_store_data(args[-1])
        reset_nclicks = 0
        if reset_button:
            reset_nclicks = args[-2] or 0
            prior_reset_nclicks = store_data.get('reset_nclicks', 0)
            if reset_nclicks != prior_reset_nclicks:
                store_data['reset_nclicks'] = reset_nclicks
                store_data['streams'] = copy.deepcopy(initial_stream_contents)
                selected_dicts = [None for _ in selected_dicts]
                relayout_dicts = [None for _ in relayout_dicts]
                any_change = True
        if store_data is None:
            store_data = {'streams': {}}
        store_data.setdefault('kdims', {})
        for i, kdim in zip(range(num_figs * 2, num_figs * 2 + len(all_kdims)), all_kdims):
            if kdim not in store_data['kdims'] or store_data['kdims'][kdim] != args[i]:
                store_data['kdims'][kdim] = args[i]
                any_change = True
        for fig_ind in range(len(initial_fig_dicts)):
            graph_id = graph_ids[fig_ind]
            for plotly_stream_type, uid_to_streams_for_type in uid_to_stream_ids.items():
                for panel_prop in plotly_stream_type.callback_properties:
                    if panel_prop == 'selected_data':
                        if graph_id + '.selectedData' in triggered_prop_ids:
                            stream_event_data = plotly_stream_type.get_event_data_from_property_update(panel_prop, selected_dicts[fig_ind], initial_fig_dicts[fig_ind])
                            any_change = update_stream_values_for_type(store_data, stream_event_data, uid_to_streams_for_type) or any_change
                    elif panel_prop == 'viewport':
                        if graph_id + '.relayoutData' in triggered_prop_ids:
                            stream_event_data = plotly_stream_type.get_event_data_from_property_update(panel_prop, relayout_dicts[fig_ind], initial_fig_dicts[fig_ind])
                            stream_event_data = {uid: event_data for uid, event_data in stream_event_data.items() if event_data['x_range'] is not None or event_data['y_range'] is not None}
                            any_change = update_stream_values_for_type(store_data, stream_event_data, uid_to_streams_for_type) or any_change
                    elif panel_prop == 'relayout_data':
                        if graph_id + '.relayoutData' in triggered_prop_ids:
                            stream_event_data = plotly_stream_type.get_event_data_from_property_update(panel_prop, relayout_dicts[fig_ind], initial_fig_dicts[fig_ind])
                            any_change = update_stream_values_for_type(store_data, stream_event_data, uid_to_streams_for_type) or any_change
        if not any_change:
            raise PreventUpdate
        for output_id in reversed(stream_callbacks):
            stream_callback = stream_callbacks[output_id]
            input_ids = stream_callback.input_ids
            fn = stream_callback.fn
            output_id = stream_callback.output_id
            input_values = [store_data['streams'][input_id] for input_id in input_ids]
            output_value = fn(*input_values)
            store_data['streams'][output_id] = output_value
        figs = [None] * num_figs
        for fig_ind, (fn, stream_ids) in fig_to_fn_stream_ids.items():
            fig_kdim_values = [store_data['kdims'][kd] for kd in kdims_per_fig[fig_ind]]
            stream_values = [store_data['streams'][stream_id] for stream_id in stream_ids]
            hvobj = fn(*fig_kdim_values + stream_values)
            plot = PlotlyRenderer.get_plot(hvobj)
            fig = plot_to_figure(plot, reset_nclicks=reset_nclicks, layout_ranges=layout_ranges[fig_ind], responsive=responsive, use_ranges=use_ranges).to_dict()
            figs[fig_ind] = fig
        return figs + [encode_store_data(store_data)]
    for i, kdim_name in enumerate(all_kdims):
        kdim_label = all_kdims[kdim_name][0]
        kdim_slider_id = kdim_name + '-' + kdim_uuids[i]
        kdim_label_id = kdim_name + '-label-' + kdim_uuids[i]

        @app.callback(Output(component_id=kdim_label_id, component_property='children'), [Input(component_id=kdim_slider_id, component_property='value')])
        def update_kdim_label(value, kdim_label=kdim_label):
            return f'{kdim_label}: {value:.2f}'
    components = DashComponents(graphs=graph_components, kdims=kdim_components, resets=reset_components, store=store, children=graph_components + list(kdim_components.values()) + reset_components + [store])
    return components