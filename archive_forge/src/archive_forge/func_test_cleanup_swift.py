import queue
from openstack.tests.functional import base
def test_cleanup_swift(self):
    if not self.user_cloud.has_service('object-store'):
        self.skipTest('Object service is requred, but not available')
    status_queue = queue.Queue()
    self.conn.object_store.create_container('test_cleanup')
    for i in range(1, 10):
        self.conn.object_store.create_object('test_cleanup', f'test{i}', data='test{i}')
    self.conn.project_cleanup(dry_run=True, wait_timeout=120, status_queue=status_queue, filters={'updated_at': '2000-01-01'})
    self.assertTrue(status_queue.empty())
    self.conn.project_cleanup(dry_run=True, wait_timeout=120, status_queue=status_queue, filters={'updated_at': '2200-01-01'})
    objects = []
    while not status_queue.empty():
        objects.append(status_queue.get())
    obj_names = list((obj.name for obj in objects))
    self.assertIn('test1', obj_names)
    obj = self.conn.object_store.get_object('test1', 'test_cleanup')
    self.assertIsNotNone(obj)
    self.conn.project_cleanup(dry_run=False, wait_timeout=600, status_queue=status_queue)
    objects.clear()
    while not status_queue.empty():
        objects.append(status_queue.get())
    self.assertIsNone(self.conn.get_container('test_container'))