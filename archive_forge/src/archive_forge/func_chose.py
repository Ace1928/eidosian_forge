import json
import logging
from bs4 import BeautifulSoup
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
def chose(self, orig_response, path, **kwargs):
    """
        Sends a HTTP GET to a url given by the present url and the given
        relative path.

        :param orig_response: The original response
        :param content: The content of the response
        :param path: The relative path to add to the base URL
        :return: The response do_click() returns
        """
    if not path.startswith('http'):
        try:
            _url = orig_response.url
        except KeyError:
            _url = kwargs['location']
        part = urlparse(_url)
        url = f'{part[0]}://{part[1]}{path}'
    else:
        url = path
    logger.info('GET %s', url)
    return self.httpc.send(url, 'GET')