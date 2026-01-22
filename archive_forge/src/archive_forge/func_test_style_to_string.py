from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_style_to_string(geom_df):
    pytest.importorskip('lxml')
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="text" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:param name="delim"><xsl:text>               </xsl:text></xsl:param>\n    <xsl:template match="/data">\n        <xsl:text>      shape  degrees  sides&#xa;</xsl:text>\n        <xsl:apply-templates select="row"/>\n    </xsl:template>\n\n    <xsl:template match="row">\n        <xsl:value-of select="concat(index, \' \',\n                                     substring($delim, 1, string-length(\'triangle\')\n                                               - string-length(shape) + 1),\n                                     shape,\n                                     substring($delim, 1, string-length(name(degrees))\n                                               - string-length(degrees) + 2),\n                                     degrees,\n                                     substring($delim, 1, string-length(name(sides))\n                                               - string-length(sides) + 2),\n                                     sides)"/>\n         <xsl:text>&#xa;</xsl:text>\n    </xsl:template>\n</xsl:stylesheet>'
    out_str = geom_df.to_string()
    out_xml = geom_df.to_xml(na_rep='NaN', stylesheet=xsl)
    assert out_xml == out_str