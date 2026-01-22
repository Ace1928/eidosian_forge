from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import container
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def template_by_name(name='OS::Barbican::GenericContainer'):
    mapping = {'OS::Barbican::GenericContainer': stack_template_generic, 'OS::Barbican::CertificateContainer': stack_template_certificate, 'OS::Barbican::RSAContainer': stack_template_rsa}
    return mapping[name]