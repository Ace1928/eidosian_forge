import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from forgeengine.memory import MemoryStore


def test_memory_save_load(tmp_path):
    path = tmp_path / "mem.json"
    store = MemoryStore(str(path))
    store.data.interactions.append({"timestamp": "t", "user": "u", "response": "r"})
    store.save()

    new_store = MemoryStore(str(path))
    assert new_store.data.interactions[0]["user"] == "u"


