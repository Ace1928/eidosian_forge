import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def create_compact_list_message(raw_elems):
    elements = [create_list_element(elem) for elem in raw_elems]
    return {'type': 'template', 'payload': {'template_type': 'list', 'top_element_style': 'COMPACT', 'elements': elements}}