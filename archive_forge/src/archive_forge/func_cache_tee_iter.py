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
def cache_tee_iter(self, image_id, image_iter, image_checksum):
    try:
        current_checksum = md5(usedforsecurity=False)
        with self.driver.open_for_write(image_id) as cache_file:
            for chunk in image_iter:
                try:
                    cache_file.write(chunk)
                finally:
                    current_checksum.update(chunk)
                    yield chunk
            cache_file.flush()
            if image_checksum and image_checksum != current_checksum.hexdigest():
                msg = _("Checksum verification failed. Aborted caching of image '%s'.") % image_id
                raise exception.GlanceException(msg)
    except exception.GlanceException as e:
        with excutils.save_and_reraise_exception():
            LOG.exception(encodeutils.exception_to_unicode(e))
    except Exception as e:
        LOG.exception(_LE("Exception encountered while tee'ing image '%(image_id)s' into cache: %(error)s. Continuing with response."), {'image_id': image_id, 'error': encodeutils.exception_to_unicode(e)})
        for chunk in image_iter:
            yield chunk