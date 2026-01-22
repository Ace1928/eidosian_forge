import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from keystoneauth1 import loading
from keystoneauth1.tests.unit.loading import utils
Test external options that are deprecated by Adapter options.

        Not to be confused with ConfLoadingDeprecatedTests, which tests conf
        options in Adapter which are themselves deprecated.
        