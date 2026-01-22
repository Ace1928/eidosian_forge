import os
import tempfile
from io import BytesIO
from urllib.request import urlopen
from Bio.KEGG.KGML.KGML_pathway import Pathway
def get_temp_imagefilename(url):
    """Return filename of temporary file containing downloaded image.

    Create a new temporary file to hold the image file at the passed URL
    and return the filename.
    """
    img = urlopen(url).read()
    im = Image.open(BytesIO(img))
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fname = f.name
    f.close()
    im.save(fname, 'PNG')
    return fname