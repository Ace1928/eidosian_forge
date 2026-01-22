from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_sync_k8s(required: bool=True):
    """
    Ensures that `kubernetes` is availableable
    """
    global _k8_available
    global SyncClient, SyncConfig, SyncStream, SyncWatch, SyncUtils, SyncType, SyncStreamFunc
    if not _k8_available:
        resolve_missing('kubernetes', required=required)
        import kubernetes.client as SyncClient
        import kubernetes.config as SyncConfig
        import kubernetes.stream.ws_client as SyncStream
        import kubernetes.stream.stream as SyncStreamFunc
        import kubernetes.watch as SyncWatch
        import kubernetes.utils as SyncUtils
        import kubernetes.client.models as SyncType
        _k8_available = True