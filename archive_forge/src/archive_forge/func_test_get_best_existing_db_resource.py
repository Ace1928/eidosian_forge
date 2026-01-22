from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_get_best_existing_db_resource(self, mock_cr):
    stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
    stack.prev_raw_template_id = 2
    stack.t.id = 3

    def db_resource(current_template_id, created_at=None, updated_at=None):
        db_res = resource_objects.Resource(stack.context)
        db_res['id'] = current_template_id
        db_res['name'] = 'A'
        db_res['current_template_id'] = current_template_id
        db_res['action'] = 'UPDATE' if updated_at else 'CREATE'
        db_res['status'] = 'COMPLETE'
        db_res['updated_at'] = updated_at
        db_res['created_at'] = created_at
        db_res['replaced_by'] = None
        return db_res
    start_time = datetime.utcfromtimestamp(0)

    def t(minutes):
        return start_time + timedelta(minutes=minutes)
    a_res_2 = db_resource(2)
    a_res_3 = db_resource(3)
    a_res_0 = db_resource(0, created_at=t(0), updated_at=t(1))
    a_res_1 = db_resource(1, created_at=t(2))
    existing_res = {a_res_2.id: a_res_2, a_res_3.id: a_res_3, a_res_0.id: a_res_0, a_res_1.id: a_res_1}
    stack.ext_rsrcs_db = existing_res
    best_res = stack._get_best_existing_rsrc_db('A')
    self.assertEqual(a_res_3.id, best_res.id)
    del existing_res[3]
    best_res = stack._get_best_existing_rsrc_db('A')
    self.assertEqual(a_res_2.id, best_res.id)
    del existing_res[2]
    best_res = stack._get_best_existing_rsrc_db('A')
    self.assertEqual(a_res_1.id, best_res.id)
    del existing_res[1]
    best_res = stack._get_best_existing_rsrc_db('A')
    self.assertEqual(a_res_0.id, best_res.id)