import pytest
import rpy2.rinterface
import rpy2.rinterface_lib.embedded
from threading import Thread
from rpy2.rinterface import embedded
@pytest.mark.skipif(embedded.rpy2_embeddedR_isinitialized, reason='Can only be tested before R is initialized.')
def test_threading_initr_simple():
    thread = ThreadWithExceptions(target=rpy2.rinterface.initr_simple)
    with pytest.warns(UserWarning, match='R is not initialized by the main thread.\\n\\W+Its taking over SIGINT cannot be reversed here.+'):
        thread.start()
        thread.join()
    assert not thread.exception