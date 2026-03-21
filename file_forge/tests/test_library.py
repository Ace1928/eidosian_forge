from pathlib import Path

from file_forge import FileForge, FileLibraryDB


def test_index_directory_and_restore_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "docs").mkdir()
    code = root / "src" / "app.py"
    doc = root / "docs" / "guide.md"
    meta = root / "config.json"
    code.write_text("print(1)\n", encoding="utf-8")
    doc.write_text("# Guide\n\ncontinuity memory lesson\n", encoding="utf-8")
    meta.write_text('{"ok":true}\n', encoding="utf-8")

    forge = FileForge(base_path=tmp_path)
    db_path = tmp_path / "file_library.sqlite"
    result = forge.index_directory(root, db_path=db_path)

    assert result["indexed"] == 3
    db = FileLibraryDB(db_path)
    code_links = db.list_links(code)
    doc_links = db.list_links(doc)
    assert any(link["forge"] == "code_forge" for link in code_links)
    assert any(link["forge"] == "knowledge_forge" for link in doc_links)
    assert any(link["forge"] == "memory_forge" for link in doc_links)

    code.unlink()
    restored = forge.restore_indexed_file(code, db_path=db_path)
    assert restored.read_text(encoding="utf-8") == "print(1)\n"


def test_index_directory_skips_unchanged_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("alpha\n", encoding="utf-8")

    forge = FileForge(base_path=tmp_path)
    db_path = tmp_path / "file_library.sqlite"
    first = forge.index_directory(root, db_path=db_path)
    second = forge.index_directory(root, db_path=db_path)

    assert first["indexed"] == 1
    assert second["indexed"] == 0
    assert second["skipped"] == 1


def test_duplicate_relationships_are_recorded(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    a = root / "a.txt"
    b = root / "b.txt"
    a.write_text("same\n", encoding="utf-8")
    b.write_text("same\n", encoding="utf-8")

    forge = FileForge(base_path=tmp_path)
    db_path = tmp_path / "file_library.sqlite"
    forge.index_directory(root, db_path=db_path)

    db = FileLibraryDB(db_path)
    rels = db.list_relationships(a)
    assert any(rel["rel_type"] == "duplicate_of" and rel["dst_path"] == str(b.resolve()) for rel in rels)


def test_restore_directory_restores_missing_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "docs").mkdir()
    kept = root / "docs" / "kept.md"
    extra = root / "docs" / "extra.ndjson"
    kept.write_text("# kept\n", encoding="utf-8")
    extra.write_text('{"event":1}\n', encoding="utf-8")

    forge = FileForge(base_path=tmp_path)
    db_path = tmp_path / "file_library.sqlite"
    forge.index_directory(root, db_path=db_path)

    restored_root = tmp_path / "restored"
    result = forge.restore_directory(root, target_root=restored_root, db_path=db_path)

    assert result["restored"] == 2
    assert (restored_root / "docs" / "kept.md").read_text(encoding="utf-8") == "# kept\n"
    assert (restored_root / "docs" / "extra.ndjson").read_text(encoding="utf-8") == '{"event":1}\n'


def test_restore_directory_can_overwrite_existing_targets(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("alpha\n", encoding="utf-8")

    forge = FileForge(base_path=tmp_path)
    db_path = tmp_path / "file_library.sqlite"
    forge.index_directory(root, db_path=db_path)

    restored_root = tmp_path / "restored"
    restored_root.mkdir()
    (restored_root / "note.txt").write_text("stale\n", encoding="utf-8")
    result = forge.restore_directory(root, target_root=restored_root, db_path=db_path, overwrite=True)

    assert result["restored"] == 1
    assert result["overwritten_existing"] == 1
    assert (restored_root / "note.txt").read_text(encoding="utf-8") == "alpha\n"


def test_summary_reports_counts_and_doc_forge_links(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    runtime = root / "doc_forge" / "runtime" / "final_docs"
    runtime.mkdir(parents=True)
    target = runtime / "guide.md"
    target.write_text("# Guide\n", encoding="utf-8")

    forge = FileForge(base_path=tmp_path)
    db_path = tmp_path / "file_library.sqlite"
    forge.index_directory(root, db_path=db_path)

    db = FileLibraryDB(db_path)
    summary = db.summary(path_prefix=root)

    assert summary["total_files"] == 1
    assert any(row["forge"] == "doc_forge" for row in summary["by_forge"])
    assert summary["recent_files"][0]["kind"] == "document"
