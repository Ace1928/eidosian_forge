import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "machine_clone" / "build_dual_live_usb.sh"
VERIFY_SCRIPT = Path(__file__).resolve().parents[1] / "machine_clone" / "verify_dual_live_usb.sh"
REPAIR_SCRIPT = Path(__file__).resolve().parents[1] / "machine_clone" / "repair_dual_live_usb.sh"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_print_grub_config_ext4_mode() -> None:
    result = run_script("--print-grub-config", "--live-fs", "ext4")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "insmod ext2" in out
    assert "search --no-floppy --label LIVE_MULTI --set=live" in out
    assert "if [ -e ($live)/$1 ];" in out
    assert "elif [ -e ($live)/live/$1 ];" in out
    assert "set_iso_path ubuntu-live.iso" in out
    assert "set_iso_path systemrescue.iso" in out
    assert "iso-scan/filename=$isofile findiso=$isofile" in out


def test_print_grub_config_exfat_mode() -> None:
    result = run_script("--print-grub-config", "--live-fs", "exfat")
    assert result.returncode == 0, result.stderr
    assert "insmod exfat" in result.stdout


def test_reject_invalid_live_fs() -> None:
    result = run_script("--print-grub-config", "--live-fs", "ntfs")
    assert result.returncode != 0
    assert "Invalid --live-fs value" in result.stderr


def test_reject_non_numeric_sizes() -> None:
    result = run_script("--print-grub-config", "--casper-rw-size-gb", "abc")
    assert result.returncode != 0
    assert "casper-rw-size-gb must be an integer" in result.stderr


def test_reject_invalid_post_verify_value() -> None:
    result = run_script("--print-grub-config", "--post-verify", "maybe")
    assert result.returncode != 0
    assert "post-verify must be yes or no" in result.stderr


def test_verify_script_help() -> None:
    result = subprocess.run(
        ["bash", str(VERIFY_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "verify_dual_live_usb.sh --device /dev/sdX" in result.stdout


def test_repair_script_help() -> None:
    result = subprocess.run(
        ["bash", str(REPAIR_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "repair_dual_live_usb.sh" in result.stdout
