from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import importutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance.common import exception
from glance.common import utils
from glance.i18n import _, _LE, _LI, _LW
def configure_driver(self):
    """
        Configure the driver for the cache and, if it fails to configure,
        fall back to using the SQLite driver which has no odd dependencies
        """
    try:
        self.driver = self.driver_class()
        self.driver.configure()
    except exception.BadDriverConfiguration as config_err:
        driver_module = self.driver_class.__module__
        LOG.warning(_LW("Image cache driver '%(driver_module)s' failed to configure. Got error: '%(config_err)s"), {'driver_module': driver_module, 'config_err': config_err})
        LOG.info(_LI('Defaulting to SQLite driver.'))
        default_module = __name__ + '.drivers.sqlite.Driver'
        self.driver_class = importutils.import_class(default_module)
        self.driver = self.driver_class()
        self.driver.configure()