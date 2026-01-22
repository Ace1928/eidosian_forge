import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union
import srsly
from thinc.api import (
from thinc.config import ConfigValidationError
from wasabi import Printer
from ..errors import Errors
from ..schemas import ConfigSchemaPretrain
from ..tokens import Doc
from ..util import dot_to_object, load_model_from_config, registry
from .example import Example
def pretrain(config: Config, output_dir: Path, resume_path: Optional[Path]=None, epoch_resume: Optional[int]=None, use_gpu: int=-1, silent: bool=True, skip_last: bool=False):
    msg = Printer(no_print=silent)
    if config['training']['seed'] is not None:
        fix_random_seed(config['training']['seed'])
    allocator = config['training']['gpu_allocator']
    if use_gpu >= 0 and allocator:
        set_gpu_allocator(allocator)
    config['initialize']['init_tok2vec'] = None
    nlp = load_model_from_config(config)
    _config = nlp.config.interpolate()
    P = registry.resolve(_config['pretraining'], schema=ConfigSchemaPretrain)
    corpus = dot_to_object(_config, P['corpus'])
    corpus = registry.resolve({'corpus': corpus})['corpus']
    batcher = P['batcher']
    model = create_pretraining_model(nlp, P)
    optimizer = P['optimizer']
    if resume_path is not None:
        epoch_resume = _resume_model(model, resume_path, epoch_resume, silent=silent)
    else:
        epoch_resume = 0
    objective = model.attrs['loss']
    tracker = ProgressTracker(frequency=10000)
    if P['n_save_epoch']:
        msg.divider(f'Pre-training tok2vec layer - starting at epoch {epoch_resume} - saving every {P['n_save_epoch']} epoch')
    else:
        msg.divider(f'Pre-training tok2vec layer - starting at epoch {epoch_resume}')
    row_settings = {'widths': (3, 10, 10, 6, 4), 'aligns': ('r', 'r', 'r', 'r', 'r')}
    msg.row(('#', '# Words', 'Total Loss', 'Loss', 'w/s'), **row_settings)

    def _save_model(epoch, is_temp=False, is_last=False):
        is_temp_str = '.temp' if is_temp else ''
        with model.use_params(optimizer.averages):
            if is_last:
                save_path = output_dir / f'model-last.bin'
            else:
                save_path = output_dir / f'model{epoch}{is_temp_str}.bin'
            with save_path.open('wb') as file_:
                file_.write(model.get_ref('tok2vec').to_bytes())
            log = {'nr_word': tracker.nr_word, 'loss': tracker.loss, 'epoch_loss': tracker.epoch_loss, 'epoch': epoch}
            with (output_dir / 'log.jsonl').open('a') as file_:
                file_.write(srsly.json_dumps(log) + '\n')
    try:
        for epoch in range(epoch_resume, P['max_epochs']):
            for batch_id, batch in enumerate(batcher(corpus(nlp))):
                docs = ensure_docs(batch)
                loss = make_update(model, docs, optimizer, objective)
                progress = tracker.update(epoch, loss, docs)
                if progress:
                    msg.row(progress, **row_settings)
                if P['n_save_every'] and batch_id % P['n_save_every'] == 0:
                    _save_model(epoch, is_temp=True)
            if P['n_save_epoch']:
                if epoch % P['n_save_epoch'] == 0 or epoch == P['max_epochs'] - 1:
                    _save_model(epoch)
            else:
                _save_model(epoch)
            tracker.epoch_loss = 0.0
    finally:
        if not skip_last:
            _save_model(P['max_epochs'], is_last=True)