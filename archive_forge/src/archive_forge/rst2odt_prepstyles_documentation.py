from lxml import etree
import sys
import zipfile
from tempfile import mkstemp
import shutil
import os

Fix a word-processor-generated styles.odt for odtwriter use: Drop page size
specifications from styles.xml in STYLE_FILE.odt.
