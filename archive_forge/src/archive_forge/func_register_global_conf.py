from typing import Any, Dict
from triad import ParamDict
def register_global_conf(conf: Dict[str, Any], on_dup: int=ParamDict.OVERWRITE) -> None:
    """Register global Fugue configs that can be picked up by any
    Fugue execution engines as the base configs.

    :param conf: the config dictionary
    :param on_dup: see :meth:`triad.collections.dict.ParamDict.update`
      , defaults to ``ParamDict.OVERWRITE``

    .. note::

        When using ``ParamDict.THROW`` or ``on_dup``, it's transactional.
        If any key in ``conf`` is already in global config and the value
        is different from the new value, then ValueError will be thrown.

    .. admonition:: Examples

        .. code-block:: python

            from fugue import register_global_conf, NativeExecutionEngine

            register_global_conf({"my.value",1})

            engine = NativeExecutionEngine()
            assert 1 == engine.conf["my.value"]

            engine = NativeExecutionEngine({"my.value",2})
            assert 2 == engine.conf["my.value"]
    """
    if on_dup == ParamDict.THROW:
        for k, v in conf.items():
            if k in _FUGUE_GLOBAL_CONF:
                vv = _FUGUE_GLOBAL_CONF[k]
                if vv != v:
                    raise ValueError(f"for global config {k}, the existed value is {vv} and can't take new value {v}")
        on_dup = ParamDict.OVERWRITE
    _FUGUE_GLOBAL_CONF.update(conf, on_dup=on_dup, deep=True)