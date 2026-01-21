"""Root package bridge for Eidosian layout."""
from pkgutil import extend_path
from pathlib import Path
__path__ = extend_path(__path__, __name__)
_root = Path(__file__).resolve().parent
_src = _root / "src" / __name__
if _src.exists():
    __path__.append(str(_src))
_nested = _root / __name__
if _nested.exists():
    __path__.append(str(_nested))
