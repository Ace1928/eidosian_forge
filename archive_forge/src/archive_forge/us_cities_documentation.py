from __future__ import annotations
import logging # isort:skip
from ..util.sampledata import external_path, load_json
 Locations of US cities with more than 5000 residents.

License: `CC BY 2.0`_

Sourced from http://www.geonames.org/export/ (subset of *cities5000.zip*)

This module contains one dict: ``data``.

.. code-block:: python

    data['lat']  # list of float
    data['lon']  # list of float

.. bokeh-sampledata-xref:: us_cities
