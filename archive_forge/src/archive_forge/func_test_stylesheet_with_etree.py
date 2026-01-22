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
def test_stylesheet_with_etree(geom_df):
    xsl = '<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">\n    <xsl:output method="xml" encoding="utf-8" indent="yes" />\n    <xsl:strip-space elements="*"/>\n\n    <xsl:template match="@*|node(*)">\n        <xsl:copy>\n            <xsl:apply-templates select="@*|node()"/>\n        </xsl:copy>\n    </xsl:template>'
    with pytest.raises(ValueError, match='To use stylesheet, you need lxml installed'):
        geom_df.to_xml(parser='etree', stylesheet=xsl)