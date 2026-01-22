import json
from django import template
from django.template.context import Context
from .base import InclusionAdminNode
@register.tag(name='submit_row')
def submit_row_tag(parser, token):
    return InclusionAdminNode(parser, token, func=submit_row, template_name='submit_line.html')