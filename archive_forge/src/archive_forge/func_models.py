from fastapi import APIRouter, HTTPException, status
from utils.rwkv import AbstractRWKV
import global_var
@router.get('/v1/models', tags=['MISC'])
@router.get('/models', tags=['MISC'])
def models():
    model: AbstractRWKV = global_var.get(global_var.Model)
    model_name = model.name if model else 'rwkv'
    return {'object': 'list', 'data': [{'id': model_name, 'object': 'model', 'owned_by': 'rwkv', 'root': model_name, 'parent': None}, *fake_models]}