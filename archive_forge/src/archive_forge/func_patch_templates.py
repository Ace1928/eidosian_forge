from django.template import TemplateSyntaxError
from django.utils.safestring import mark_safe
from django import VERSION as DJANGO_VERSION
from sentry_sdk import _functools, Hub
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
def patch_templates():
    from django.template.response import SimpleTemplateResponse
    from sentry_sdk.integrations.django import DjangoIntegration
    real_rendered_content = SimpleTemplateResponse.rendered_content

    @property
    def rendered_content(self):
        hub = Hub.current
        if hub.get_integration(DjangoIntegration) is None:
            return real_rendered_content.fget(self)
        with hub.start_span(op=OP.TEMPLATE_RENDER, description=_get_template_name_description(self.template_name)) as span:
            span.set_data('context', self.context_data)
            return real_rendered_content.fget(self)
    SimpleTemplateResponse.rendered_content = rendered_content
    if DJANGO_VERSION < (1, 7):
        return
    import django.shortcuts
    real_render = django.shortcuts.render

    @_functools.wraps(real_render)
    def render(request, template_name, context=None, *args, **kwargs):
        hub = Hub.current
        if hub.get_integration(DjangoIntegration) is None:
            return real_render(request, template_name, context, *args, **kwargs)
        context = context or {}
        if 'sentry_trace_meta' not in context:
            context['sentry_trace_meta'] = mark_safe(hub.trace_propagation_meta())
        with hub.start_span(op=OP.TEMPLATE_RENDER, description=_get_template_name_description(template_name)) as span:
            span.set_data('context', context)
            return real_render(request, template_name, context, *args, **kwargs)
    django.shortcuts.render = render