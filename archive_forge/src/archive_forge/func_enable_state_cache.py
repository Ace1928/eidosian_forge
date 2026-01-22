from typing import Any, Dict, List, Union
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
import gc
import copy
import global_var
@router.post('/enable-state-cache', tags=['State Cache'])
def enable_state_cache():
    global trie, dtrie
    if global_var.get(global_var.Deploy_Mode) is True:
        raise HTTPException(status.HTTP_403_FORBIDDEN)
    try:
        import cyac
        trie = cyac.Trie()
        dtrie = {}
        gc.collect()
        print('state cache enabled')
        return 'success'
    except ModuleNotFoundError:
        print('state cache disabled')
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'cyac not found')