import datetime
import mock
import pytest  # type: ignore
from google.auth import app_engine
@mock.patch.object(app_engine, 'app_identity', new=None)
def test_get_project_id_missing_apis():
    with pytest.raises(EnvironmentError) as excinfo:
        assert app_engine.get_project_id()
    assert excinfo.match('App Engine APIs are not available')