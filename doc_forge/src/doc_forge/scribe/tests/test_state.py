import json
from doc_forge.scribe.state import ProcessorState, atomic_write_json

def test_atomic_write(temp_forge_root):
    p = temp_forge_root / "test.json"
    data = {"foo": "bar"}
    atomic_write_json(p, data)
    assert json.loads(p.read_text()) == data
    assert not p.with_suffix(".json.tmp").exists()

def test_state_persistence(temp_forge_root):
    state_p = temp_forge_root / "state.json"
    status_p = temp_forge_root / "status.json"
    index_p = temp_forge_root / "index.json"
    
    state = ProcessorState(state_p, status_p, index_p)
    state.update("processed", 10)
    state.persist()
    
    loaded = json.loads(state_p.read_text())
    assert loaded["processed"] == 10
    
    status = json.loads(status_p.read_text())
    assert status["processed"] == 10
