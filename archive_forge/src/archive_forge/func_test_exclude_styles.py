from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_exclude_styles(styler):
    result = styler.to_html(exclude_styles=True, doctype_html=True)
    expected = dedent('        <!DOCTYPE html>\n        <html>\n        <head>\n        <meta charset="utf-8">\n        </head>\n        <body>\n        <table>\n          <thead>\n            <tr>\n              <th >&nbsp;</th>\n              <th >A</th>\n            </tr>\n          </thead>\n          <tbody>\n            <tr>\n              <th >a</th>\n              <td >2.610000</td>\n            </tr>\n            <tr>\n              <th >b</th>\n              <td >2.690000</td>\n            </tr>\n          </tbody>\n        </table>\n        </body>\n        </html>\n        ')
    assert result == expected