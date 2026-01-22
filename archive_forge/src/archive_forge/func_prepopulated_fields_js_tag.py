import json
from django import template
from django.template.context import Context
from .base import InclusionAdminNode
@register.tag(name='prepopulated_fields_js')
def prepopulated_fields_js_tag(parser, token):
    return InclusionAdminNode(parser, token, func=prepopulated_fields_js, template_name='prepopulated_fields_js.html')