import time
def eval_cb(self, param):
    self._process_batch(param, 'eval')
    self._do_update()