import threading
from unittest import mock
import greenlet
from oslo_config import cfg
from oslotest import base
from oslo_reports.generators import conf as os_cgen
from oslo_reports.generators import threading as os_tgen
from oslo_reports.generators import version as os_pgen
from oslo_reports.models import threading as os_tmod
def test_package_report_generator(self):

    class VersionObj(object):

        def vendor_string(self):
            return 'Cheese Shoppe'

        def product_string(self):
            return 'Sharp Cheddar'

        def version_string_with_package(self):
            return '1.0.0'
    model = os_pgen.PackageReportGenerator(VersionObj())()
    model.set_current_view_type('text')
    target_str = 'product = Sharp Cheddar\nvendor = Cheese Shoppe\nversion = 1.0.0'
    self.assertEqual(target_str, str(model))