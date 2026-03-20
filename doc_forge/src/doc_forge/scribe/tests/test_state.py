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
    history_p = temp_forge_root / "history.jsonl"

    state = ProcessorState(state_p, status_p, index_p, history_p)
    state.update("processed", 10)
    state.update("phase", "processing")
    state.persist()

    loaded = json.loads(state_p.read_text())
    assert loaded["processed"] == 10

    status = json.loads(status_p.read_text())
    assert status["contract"] == "eidos.doc_processor.status.v1"
    assert status["processed"] == 10
    assert status["phase"] == "processing"
    history_rows = history_p.read_text().splitlines()
    assert len(history_rows) == 1


def test_state_seeds_history_from_existing_status(temp_forge_root):
    state_p = temp_forge_root / "state.json"
    status_p = temp_forge_root / "status.json"
    index_p = temp_forge_root / "index.json"
    history_p = temp_forge_root / "history.jsonl"
    status_p.write_text(json.dumps({"status": "idle", "phase": "seeded"}), encoding="utf-8")

    ProcessorState(state_p, status_p, index_p, history_p)

    seeded = history_p.read_text(encoding="utf-8").splitlines()
    assert len(seeded) == 1
    assert json.loads(seeded[0])["phase"] == "seeded"
