import base64
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from io import BytesIO
from xml.dom import minidom
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
def get_namespace_id(doc, url):
    attributes = doc.documentElement.attributes
    namespace = ''
    for i in range(attributes.length):
        if attributes.item(i).value == url:
            if ':' in attributes.item(i).name:
                namespace = attributes.item(i).name.split(':')[1] + ':'
                break
    return namespace