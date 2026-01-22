import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree
import xml.parsers.expat
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib import font_manager as fm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
def test_svg_clear_default_metadata(monkeypatch):
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')
    metadata_contains = {'creator': mpl.__version__, 'date': '1970-08-16', 'format': 'image/svg+xml', 'type': 'StillImage'}
    SVGNS = '{http://www.w3.org/2000/svg}'
    RDFNS = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
    CCNS = '{http://creativecommons.org/ns#}'
    DCNS = '{http://purl.org/dc/elements/1.1/}'
    fig, ax = plt.subplots()
    for name in metadata_contains:
        with BytesIO() as fd:
            fig.savefig(fd, format='svg', metadata={name.title(): None})
            buf = fd.getvalue().decode()
        root = xml.etree.ElementTree.fromstring(buf)
        work, = root.findall(f'./{SVGNS}metadata/{RDFNS}RDF/{CCNS}Work')
        for key in metadata_contains:
            data = work.findall(f'./{DCNS}{key}')
            if key == name:
                assert not data
                continue
            data, = data
            xmlstr = xml.etree.ElementTree.tostring(data, encoding='unicode')
            assert metadata_contains[key] in xmlstr