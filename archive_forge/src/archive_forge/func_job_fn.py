import json
import traceback
from contextvars import copy_context
from _plotly_utils.utils import PlotlyJSONEncoder
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
from dash.long_callback.managers import BaseLongCallbackManager
@celery_app.task(name=f'long_callback_{key}')
def job_fn(result_key, progress_key, user_callback_args, context=None):

    def _set_progress(progress_value):
        if not isinstance(progress_value, (list, tuple)):
            progress_value = [progress_value]
        cache.set(progress_key, json.dumps(progress_value, cls=PlotlyJSONEncoder))
    maybe_progress = [_set_progress] if progress else []
    ctx = copy_context()

    def run():
        c = AttributeDict(**context)
        c.ignore_register_page = False
        context_value.set(c)
        try:
            if isinstance(user_callback_args, dict):
                user_callback_output = fn(*maybe_progress, **user_callback_args)
            elif isinstance(user_callback_args, (list, tuple)):
                user_callback_output = fn(*maybe_progress, *user_callback_args)
            else:
                user_callback_output = fn(*maybe_progress, user_callback_args)
        except PreventUpdate:
            cache.set(result_key, json.dumps({'_dash_no_update': '_dash_no_update'}, cls=PlotlyJSONEncoder))
        except Exception as err:
            cache.set(result_key, json.dumps({'long_callback_error': {'msg': str(err), 'tb': traceback.format_exc()}}))
        else:
            cache.set(result_key, json.dumps(user_callback_output, cls=PlotlyJSONEncoder))
    ctx.run(run)