from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_k8s(is_sync: bool=False, is_async: bool=True, is_operator: bool=False, required: bool=True):
    """
    Ensures that `kubernetes`, `kubernetes_asyncio`, `kopf` are availableable
    """
    if is_operator:
        resolve_kopf(required=required)
    if is_async:
        resolve_async_k8s(required=required)
        resolve_aiocache(required=required)
    if is_sync or not is_async:
        resolve_sync_k8s(required=required)