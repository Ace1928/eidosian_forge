import tempfile
from typing import Iterable, Dict, List
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work import _MeasurementSpec, BitstringAccumulator, group_settings_greedy, InitObsSetting
from cirq.work.observable_measurement import (
def test_checkpoint_options():
    assert _parse_checkpoint_options(False, None, None) == (None, None)
    with pytest.raises(ValueError):
        _parse_checkpoint_options(False, 'test', None)
    with pytest.raises(ValueError):
        _parse_checkpoint_options(False, None, 'test')
    with pytest.raises(ValueError):
        _parse_checkpoint_options(False, 'test1', 'test2')
    chk, chkprev = _parse_checkpoint_options(True, None, None)
    assert chk.startswith(tempfile.gettempdir())
    assert chk.endswith('observables.json')
    assert chkprev.startswith(tempfile.gettempdir())
    assert chkprev.endswith('observables.prev.json')
    chk, chkprev = _parse_checkpoint_options(True, None, 'prev.json')
    assert chk.startswith(tempfile.gettempdir())
    assert chk.endswith('observables.json')
    assert chkprev == 'prev.json'
    chk, chkprev = _parse_checkpoint_options(True, 'my_fancy_observables.json', None)
    assert chk == 'my_fancy_observables.json'
    assert chkprev == 'my_fancy_observables.prev.json'
    chk, chkprev = _parse_checkpoint_options(True, 'my_fancy/observables.json', None)
    assert chk == 'my_fancy/observables.json'
    assert chkprev == 'my_fancy/observables.prev.json'
    with pytest.raises(ValueError, match='Please use a `.json` filename.*'):
        _parse_checkpoint_options(True, 'my_fancy_observables.obs', None)
    with pytest.raises(ValueError, match="pattern of 'filename.extension'.*"):
        _parse_checkpoint_options(True, 'my_fancy_observables', None)
    with pytest.raises(ValueError, match="pattern of 'filename.extension'.*"):
        _parse_checkpoint_options(True, '.obs', None)
    with pytest.raises(ValueError, match="pattern of 'filename.extension'.*"):
        _parse_checkpoint_options(True, 'obs.', None)
    with pytest.raises(ValueError, match="pattern of 'filename.extension'.*"):
        _parse_checkpoint_options(True, '', None)
    chk, chkprev = _parse_checkpoint_options(True, 'test1', 'test2')
    assert chk == 'test1'
    assert chkprev == 'test2'