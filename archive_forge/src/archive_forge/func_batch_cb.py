import time
def batch_cb(self, param):
    if param.nbatch % self.frequent == 0:
        self._process_batch(param, 'train')
    if self.interval_elapsed():
        self._do_update()