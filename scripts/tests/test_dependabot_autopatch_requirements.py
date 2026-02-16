from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "dependabot_autopatch_requirements.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("dependabot_autopatch_requirements", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("dependabot_autopatch_requirements", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


autopatch_mod = _load_module()


def _alert(
    *,
    number: int,
    state: str,
    severity: str,
    ecosystem: str,
    package: str,
    manifest: str,
    patched: str,
) -> dict:
    return {
        "number": number,
        "state": state,
        "dependency": {
            "manifest_path": manifest,
            "package": {
                "ecosystem": ecosystem,
                "name": package,
            },
        },
        "security_vulnerability": {
            "severity": severity,
            "first_patched_version": {"identifier": patched},
        },
    }


def test_collect_patch_targets_uses_highest_patched_version() -> None:
    alerts = [
        _alert(
            number=1,
            state="open",
            severity="high",
            ecosystem="pip",
            package="urllib3",
            manifest="requirements.txt",
            patched="2.5.0",
        ),
        _alert(
            number=2,
            state="open",
            severity="critical",
            ecosystem="pip",
            package="urllib3",
            manifest="requirements.txt",
            patched="2.6.3",
        ),
        _alert(
            number=3,
            state="dismissed",
            severity="critical",
            ecosystem="pip",
            package="jinja2",
            manifest="requirements.txt",
            patched="3.1.6",
        ),
    ]
    targets = autopatch_mod.collect_patch_targets(
        alerts=alerts,
        allowed_states={"open"},
        allowed_severities={"critical", "high"},
        allowed_ecosystems={"pip"},
        manifest_allowlist=None,
    )
    assert "requirements.txt" in targets
    urllib3 = targets["requirements.txt"]["urllib3"]
    assert urllib3.target_version == "2.6.3"
    assert urllib3.severity == "critical"
    assert urllib3.alert_numbers == (1, 2)
    assert "jinja2" not in targets["requirements.txt"]


def test_patch_requirements_file_updates_pinned_and_tracks_unresolved(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text(
        "requests==2.30.0\n"
        "urllib3>=1.26.0\n"
        "Jinja2==3.1.0 ; python_version >= '3.10'  # keep me\n",
        encoding="utf-8",
    )
    targets = {
        "requests": autopatch_mod.PatchTarget(
            package="requests",
            manifest_path="requirements.txt",
            target_version="2.32.5",
            severity="high",
            alert_numbers=(11,),
        ),
        "urllib3": autopatch_mod.PatchTarget(
            package="urllib3",
            manifest_path="requirements.txt",
            target_version="2.6.3",
            severity="critical",
            alert_numbers=(12,),
        ),
        "jinja2": autopatch_mod.PatchTarget(
            package="jinja2",
            manifest_path="requirements.txt",
            target_version="3.1.6",
            severity="high",
            alert_numbers=(13,),
        ),
        "pillow": autopatch_mod.PatchTarget(
            package="pillow",
            manifest_path="requirements.txt",
            target_version="12.1.1",
            severity="high",
            alert_numbers=(14,),
        ),
    }
    patched, summary = autopatch_mod.patch_requirements_file(req, targets)

    assert "requests==2.32.5" in patched
    assert "Jinja2==3.1.6 ; python_version >= '3.10'  # keep me" in patched
    assert summary["updated_count"] == 2
    assert "urllib3" in summary["unresolved_unpinned"]
    assert "pillow" in summary["unresolved_missing"]


def test_apply_patch_targets_write_mode_creates_backup(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    manifest = repo / "requirements.txt"
    manifest.write_text("urllib3==2.0.0\n", encoding="utf-8")

    patch_targets = {
        "requirements.txt": {
            "urllib3": autopatch_mod.PatchTarget(
                package="urllib3",
                manifest_path="requirements.txt",
                target_version="2.6.3",
                severity="critical",
                alert_numbers=(99,),
            )
        }
    }
    backups = tmp_path / "backups"
    report = autopatch_mod.apply_patch_targets(
        repo_root=repo,
        patch_targets=patch_targets,
        write=True,
        backup_root=backups,
    )
    assert report["totals"]["updated_requirements_count"] == 1
    assert report["totals"]["changed_manifest_count"] == 1
    assert manifest.read_text(encoding="utf-8") == "urllib3==2.6.3\n"
    backup_dir = Path(report["backup_dir"])
    assert backup_dir.exists()
    backup_file = backup_dir / "requirements.txt"
    assert backup_file.exists()
    assert backup_file.read_text(encoding="utf-8") == "urllib3==2.0.0\n"
