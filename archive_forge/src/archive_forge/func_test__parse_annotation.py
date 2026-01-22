from adagio.shells.interfaceless import function_to_taskspec, _get_origin_type, _parse_annotation
from typing import Any, Dict, List, Union, Optional, Tuple
from pytest import raises
import inspect
def test__parse_annotation():
    assert dict(data_type=object, nullable=True) == _parse_annotation(None)
    assert dict(data_type=object, nullable=True) == _parse_annotation(inspect.Parameter.empty)
    assert dict(data_type=object, nullable=True) == _parse_annotation(Any)
    assert dict(data_type=int, nullable=False) == _parse_annotation(int)
    assert dict(data_type=dict, nullable=False) == _parse_annotation(Dict[str, Any])
    assert dict(data_type=object, nullable=True) == _parse_annotation(Optional[Any])
    assert dict(data_type=str, nullable=True) == _parse_annotation(Optional[str])
    assert dict(data_type=dict, nullable=True) == _parse_annotation(Optional[Dict[str, Any]])
    assert dict(data_type=dict, nullable=True) == _parse_annotation(Union[None, Dict[str, Any]])
    assert dict(data_type=dict, nullable=True) == _parse_annotation(Union[Dict[str, Any], None])
    assert dict(data_type=dict, nullable=True) == _parse_annotation(Union[Dict[str, Any], None, None])
    assert dict(data_type=dict, nullable=False) == _parse_annotation(Union[Dict[str, Any]])
    raises(TypeError, lambda: _parse_annotation(Union[Dict[str, Any], List[str]]))
    raises(TypeError, lambda: _parse_annotation(Union[Dict[str, Any], List[str], None]))
    raises(TypeError, lambda: _parse_annotation(Union[None]))
    raises(TypeError, lambda: _parse_annotation(Union[None, None]))
    raises(TypeError, lambda: _parse_annotation(type(None)))