import os
import re
from ... import exc
from ...engine import url as sa_url
from ...testing.provision import create_db
from ...testing.provision import drop_db
from ...testing.provision import follower_url_from_main
from ...testing.provision import generate_driver_url
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import run_reap_dbs
from ...testing.provision import stop_test_class_outside_fixtures
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import upsert
@generate_driver_url.for_db('sqlite')
def generate_driver_url(url, driver, query_str):
    url = _format_url(url, driver, None)
    try:
        url.get_dialect()
    except exc.NoSuchModuleError:
        return None
    else:
        return url