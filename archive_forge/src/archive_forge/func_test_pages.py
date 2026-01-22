import os
import pytest
import rpy2.robjects as robjects
import rpy2.robjects.help as rh
@pytest.mark.xfail(os.name == 'nt', reason='Windows is missing library/translations/Meta/Rd.rds file')
def test_pages():
    pages = rh.pages('plot')
    assert isinstance(pages, tuple)
    assert all((isinstance(elt, rh.Page) for elt in pages))