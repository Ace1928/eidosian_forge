from sentry_sdk import consts
from sentry_sdk._types import TYPE_CHECKING
import sentry_sdk
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import logger, capture_internal_exceptions, event_from_exception
@wraps(f)
def new_embeddings_create(*args, **kwargs):
    hub = Hub.current
    if not hub:
        return f(*args, **kwargs)
    integration = hub.get_integration(OpenAIIntegration)
    if not integration:
        return f(*args, **kwargs)
    with sentry_sdk.start_span(op=consts.OP.OPENAI_EMBEDDINGS_CREATE, description='OpenAI Embedding Creation') as span:
        if 'input' in kwargs and (_should_send_default_pii() and integration.include_prompts):
            if isinstance(kwargs['input'], str):
                set_data_normalized(span, 'ai.input_messages', [kwargs['input']])
            elif isinstance(kwargs['input'], list) and len(kwargs['input']) > 0 and isinstance(kwargs['input'][0], str):
                set_data_normalized(span, 'ai.input_messages', kwargs['input'])
        if 'model' in kwargs:
            set_data_normalized(span, 'ai.model_id', kwargs['model'])
        try:
            response = f(*args, **kwargs)
        except Exception as e:
            _capture_exception(Hub.current, e)
            raise e from None
        prompt_tokens = 0
        total_tokens = 0
        if hasattr(response, 'usage'):
            if hasattr(response.usage, 'prompt_tokens') and isinstance(response.usage.prompt_tokens, int):
                prompt_tokens = response.usage.prompt_tokens
            if hasattr(response.usage, 'total_tokens') and isinstance(response.usage.total_tokens, int):
                total_tokens = response.usage.total_tokens
        if prompt_tokens == 0:
            prompt_tokens = count_tokens(kwargs['input'] or '')
        if total_tokens == 0:
            total_tokens = prompt_tokens
        set_data_normalized(span, PROMPT_TOKENS_USED, prompt_tokens)
        set_data_normalized(span, TOTAL_TOKENS_USED, total_tokens)
        return response