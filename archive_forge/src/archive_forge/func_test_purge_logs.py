import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.parametrize('n', [0, 1, 5])
def test_purge_logs(tmp_path, file_handler, n):
    from kivy.config import Config
    from kivy.logger import FileHandler
    Config.set('kivy', 'log_dir', str(tmp_path))
    Config.set('kivy', 'log_maxfiles', n)
    handler = FileHandler()
    handler._configure()
    open_file = pathlib.Path(handler.filename).name
    time.sleep(0.05)
    names = [f'log_{i}.txt' for i in range(n + 2)]
    for name in names:
        p = tmp_path / name
        p.write_text('some data')
        time.sleep(0.05)
    handler.purge_logs()
    expected_names = list(reversed(names))[:n]
    files = {f.name for f in tmp_path.iterdir()}
    if open_file in files:
        files.remove(open_file)
        if len(expected_names) == len(files) + 1:
            expected_names = expected_names[:-1]
    assert set(expected_names) == files