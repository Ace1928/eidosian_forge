from urllib.parse import quote
from django.http import (
from django.template import Context, Engine, TemplateDoesNotExist, loader
from django.views.decorators.csrf import requires_csrf_token
@requires_csrf_token
def bad_request(request, exception, template_name=ERROR_400_TEMPLATE_NAME):
    """
    400 error handler.

    Templates: :template:`400.html`
    Context: None
    """
    try:
        template = loader.get_template(template_name)
    except TemplateDoesNotExist:
        if template_name != ERROR_400_TEMPLATE_NAME:
            raise
        return HttpResponseBadRequest(ERROR_PAGE_TEMPLATE % {'title': 'Bad Request (400)', 'details': ''})
    return HttpResponseBadRequest(template.render())