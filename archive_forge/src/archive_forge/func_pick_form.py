import json
import logging
from bs4 import BeautifulSoup
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
def pick_form(self, response, url=None, **kwargs):
    """
        Picks which form in a web-page that should be used

        :param response: A HTTP request response. A DResponse instance
        :param content: The HTTP response content
        :param url: The url the request was sent to
        :param kwargs: Extra key word arguments
        :return: The picked form or None of no form matched the criteria.
        """
    forms = ParseResponseEx(response)
    if not forms:
        raise FlowException(content=response.text, url=url)
    _form = None
    forms = forms[1:]
    if len(forms) == 1:
        _form = forms[0]
    elif 'pick' in kwargs:
        _dict = kwargs['pick']
        for form in forms:
            if _form:
                break
            for key, _ava in _dict.items():
                if key == 'form':
                    _keys = form.attrs.keys()
                    for attr, val in _ava.items():
                        if attr in _keys and val == form.attrs[attr]:
                            _form = form
                elif key == 'control':
                    prop = _ava['id']
                    _default = _ava['value']
                    try:
                        orig_val = form[prop]
                        if isinstance(orig_val, str):
                            if orig_val == _default:
                                _form = form
                        elif _default in orig_val:
                            _form = form
                    except KeyError:
                        pass
                    except ControlNotFoundError:
                        pass
                elif key == 'method':
                    if form.method == _ava:
                        _form = form
                else:
                    _form = None
                if not _form:
                    break
    elif 'index' in kwargs:
        _form = forms[int(kwargs['index'])]
    return _form