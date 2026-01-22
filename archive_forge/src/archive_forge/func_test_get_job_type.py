from unittest import mock
import uuid
from saharaclient.api import base as sahara_base
from heat.common import exception
from heat.engine.clients.os import sahara
from heat.tests import common
from heat.tests import utils
def test_get_job_type(self):
    """Tests the get_job_type function."""
    job_type = 'myfakejobtype'
    self.my_jobtype = job_type

    def side_effect(name):
        if name == job_type:
            return self.my_jobtype
        else:
            raise sahara_base.APIException(error_code=404, error_name='NOT_FOUND')
    self.sahara_client.job_types.find_unique.side_effect = side_effect
    self.assertEqual(self.sahara_plugin.get_job_type(job_type), job_type)
    self.assertRaises(exception.EntityNotFound, self.sahara_plugin.get_job_type, 'nojobtype')
    calls = [mock.call(name=job_type), mock.call(name='nojobtype')]
    self.sahara_client.job_types.find_unique.assert_has_calls(calls)