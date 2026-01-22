import copy
from . import ElementTree
from urllib.parse import urljoin
def default_loader(href, parse, encoding=None):
    if parse == 'xml':
        with open(href, 'rb') as file:
            data = ElementTree.parse(file).getroot()
    else:
        if not encoding:
            encoding = 'UTF-8'
        with open(href, 'r', encoding=encoding) as file:
            data = file.read()
    return data