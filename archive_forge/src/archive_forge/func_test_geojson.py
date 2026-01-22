import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
def test_geojson():
    gj = display.GeoJSON(data={'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-81.327, 296.038]}, 'properties': {'name': 'Inca City'}}, url_template='http://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/{basemap_id}/{z}/{x}/{y}.png', layer_options={'basemap_id': 'celestia_mars-shaded-16k_global', 'attribution': 'Celestia/praesepe', 'minZoom': 0, 'maxZoom': 18})
    assert '<IPython.core.display.GeoJSON object>' == str(gj)