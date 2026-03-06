from pathlib import Path

from eidosian_vector import HNSWVectorStore


def test_hnsw_store_roundtrip_query_and_delete(tmp_path: Path) -> None:
    store = HNSWVectorStore(tmp_path / "vectors")
    store.upsert("alpha", [1.0, 0.0, 0.0, 0.0], text="alpha text", metadata={"kind": "a"})
    store.upsert("beta", [0.0, 1.0, 0.0, 0.0], text="beta text", metadata={"kind": "b"})

    results = store.query([0.95, 0.05, 0.0, 0.0], limit=1)
    assert len(results) == 1
    assert results[0].item_id == "alpha"
    assert results[0].text == "alpha text"

    reloaded = HNSWVectorStore(tmp_path / "vectors")
    reloaded_results = reloaded.query([0.95, 0.05, 0.0, 0.0], limit=1)
    assert len(reloaded_results) == 1
    assert reloaded_results[0].item_id == "alpha"

    assert reloaded.delete("alpha")
    after_delete = reloaded.query([0.95, 0.05, 0.0, 0.0], limit=2)
    assert [item.item_id for item in after_delete] == ["beta"]
