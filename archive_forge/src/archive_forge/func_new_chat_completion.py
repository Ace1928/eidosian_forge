from sentry_sdk import consts
from sentry_sdk._types import TYPE_CHECKING
import sentry_sdk
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import logger, capture_internal_exceptions, event_from_exception
@wraps(f)
def new_chat_completion(*args, **kwargs):
    hub = Hub.current
    if not hub:
        return f(*args, **kwargs)
    integration = hub.get_integration(OpenAIIntegration)
    if not integration:
        return f(*args, **kwargs)
    if 'messages' not in kwargs:
        return f(*args, **kwargs)
    try:
        iter(kwargs['messages'])
    except TypeError:
        return f(*args, **kwargs)
    kwargs['messages'] = list(kwargs['messages'])
    messages = kwargs['messages']
    model = kwargs.get('model')
    streaming = kwargs.get('stream')
    span = sentry_sdk.start_span(op=consts.OP.OPENAI_CHAT_COMPLETIONS_CREATE, description='Chat Completion')
    span.__enter__()
    try:
        res = f(*args, **kwargs)
    except Exception as e:
        _capture_exception(Hub.current, e)
        span.__exit__(None, None, None)
        raise e from None
    with capture_internal_exceptions():
        if _should_send_default_pii() and integration.include_prompts:
            set_data_normalized(span, 'ai.input_messages', messages)
        set_data_normalized(span, 'ai.model_id', model)
        set_data_normalized(span, 'ai.streaming', streaming)
        if hasattr(res, 'choices'):
            if _should_send_default_pii() and integration.include_prompts:
                set_data_normalized(span, 'ai.responses', list(map(lambda x: x.message, res.choices)))
            _calculate_chat_completion_usage(messages, res, span)
            span.__exit__(None, None, None)
        elif hasattr(res, '_iterator'):
            data_buf: list[list[str]] = []
            old_iterator = res._iterator

            def new_iterator():
                with capture_internal_exceptions():
                    for x in old_iterator:
                        if hasattr(x, 'choices'):
                            choice_index = 0
                            for choice in x.choices:
                                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                                    content = choice.delta.content
                                    if len(data_buf) <= choice_index:
                                        data_buf.append([])
                                    data_buf[choice_index].append(content or '')
                                choice_index += 1
                        yield x
                    if len(data_buf) > 0:
                        all_responses = list(map(lambda chunk: ''.join(chunk), data_buf))
                        if _should_send_default_pii() and integration.include_prompts:
                            set_data_normalized(span, 'ai.responses', all_responses)
                        _calculate_chat_completion_usage(messages, res, span, all_responses)
                span.__exit__(None, None, None)
            res._iterator = new_iterator()
        else:
            set_data_normalized(span, 'unknown_response', True)
            span.__exit__(None, None, None)
        return res