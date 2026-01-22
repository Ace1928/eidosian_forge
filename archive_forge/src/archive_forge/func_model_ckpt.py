from ._base import *
@property
def model_ckpt(self):
    if not self.model_dir:
        return None
    if not self._model_ckpt_file:
        for ckpt in ['model.bin', 'pytorch_model.bin']:
            model_ckpt = File.join(self.model_dir, ckpt)
            if File.exists(model_ckpt):
                self._model_ckpt_file = model_ckpt
                return self._model_ckpt_file
        self._model_ckpt_file = File.join(self.model_dir, 'pytorch_model.bin')
    return self._model_ckpt_file