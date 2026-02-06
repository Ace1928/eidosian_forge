from __future__ import annotations

import json

from moltbook_forge.receipts import build_receipt, load_receipt, save_receipt


def test_receipt_sign_and_verify(tmp_path) -> None:
    receipt = build_receipt("input text", summary="summary")
    path = tmp_path / "receipt.json"
    save_receipt(str(path), receipt)
    loaded = load_receipt(str(path))
    assert loaded.verify() is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["summary"] = "tampered"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tampered = load_receipt(str(path))
    assert tampered.verify() is False
