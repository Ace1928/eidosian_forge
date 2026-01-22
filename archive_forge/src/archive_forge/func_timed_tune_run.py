from collections import Counter
import json
import numpy as np
import os
import pickle
import tempfile
import time
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.callback import Callback
from ray._private.test_utils import safe_write_to_results_json
def timed_tune_run(name: str, num_samples: int, results_per_second: int=1, trial_length_s: int=1, max_runtime: int=300, checkpoint_freq_s: int=-1, checkpoint_size_b: int=0, checkpoint_num_files: int=1, **tune_kwargs) -> bool:
    durable = 'storage_path' in tune_kwargs and tune_kwargs['storage_path'] and (tune_kwargs['storage_path'].startswith('s3://') or tune_kwargs['storage_path'].startswith('gs://'))
    sleep_time = 1.0 / results_per_second
    num_iters = int(trial_length_s / sleep_time)
    checkpoint_iters = -1
    if checkpoint_freq_s >= 0:
        checkpoint_iters = int(checkpoint_freq_s / sleep_time)
    config = {'score': tune.uniform(0.0, 1.0), 'num_iters': num_iters, 'sleep_time': sleep_time, 'checkpoint_iters': checkpoint_iters, 'checkpoint_size_b': checkpoint_size_b, 'checkpoint_num_files': checkpoint_num_files}
    print(f'Starting benchmark with config: {config}')
    run_kwargs = {'reuse_actors': True, 'verbose': 2}
    run_kwargs.update(tune_kwargs)
    _train = function_trainable
    if durable:
        _train = TestDurableTrainable
        run_kwargs['checkpoint_freq'] = checkpoint_iters
    start_time = time.monotonic()
    analysis = tune.run(_train, config=config, num_samples=num_samples, raise_on_failed_trial=False, **run_kwargs)
    time_taken = time.monotonic() - start_time
    result = {'time_taken': time_taken, 'trial_states': dict(Counter([trial.status for trial in analysis.trials])), 'last_update': time.time()}
    test_output_json = os.environ.get('TEST_OUTPUT_JSON', '/tmp/tune_test.json')
    with open(test_output_json, 'wt') as f:
        json.dump(result, f)
    success = time_taken <= max_runtime
    if not success:
        print(f'The {name} test took {time_taken:.2f} seconds, but should not have exceeded {max_runtime:.2f} seconds. Test failed. \n\n--- FAILED: {name.upper()} ::: {time_taken:.2f} > {max_runtime:.2f} ---')
    else:
        print(f'The {name} test took {time_taken:.2f} seconds, which is below the budget of {max_runtime:.2f} seconds. Test successful. \n\n--- PASSED: {name.upper()} ::: {time_taken:.2f} <= {max_runtime:.2f} ---')
    return success