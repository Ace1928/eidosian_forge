import json
import logging
from bs4 import BeautifulSoup
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
def do_click(self, form, **kwargs):
    """
        Emulates the user clicking submit on a form.

        :param form: The form that should be submitted
        :return: What do_request() returns
        """
    if 'click' in kwargs:
        request = None
        _name = kwargs['click']
        try:
            _ = form.find_control(name=_name)
            request = form.click(name=_name)
        except AmbiguityError:
            _val = kwargs['set'][_name]
            _nr = 0
            while True:
                try:
                    cntrl = form.find_control(name=_name, nr=_nr)
                    if cntrl.value == _val:
                        request = form.click(name=_name, nr=_nr)
                        break
                    else:
                        _nr += 1
                except ControlNotFoundError:
                    raise Exception(NO_CTRL % (_name, _val))
    else:
        request = form.click()
    headers = {}
    for key, val in request.unredirected_hdrs.items():
        headers[key] = val
    url = request._Request__original
    if form.method == 'POST':
        return self.httpc.send(url, 'POST', data=request.data, headers=headers)
    else:
        return self.httpc.send(url, 'GET', headers=headers)