import time
def train_cb(self, param):
    """Callback funtion for training.
        """
    if param.nbatch % self.frequent == 0:
        self._process_batch(param, 'train')