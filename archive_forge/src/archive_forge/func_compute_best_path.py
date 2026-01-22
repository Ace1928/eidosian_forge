import logging
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGP_PROCESSOR_ERROR_CODE
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.utils import circlist
from os_ken.services.protocols.bgp.utils.evtlet import EventletIOFactory
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGINATOR_ID
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_CLUSTER_LIST
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_EGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
def compute_best_path(local_asn, path1, path2):
    """Compares given paths and returns best path.

    Parameters:
        -`local_asn`: asn of local bgpspeaker
        -`path1`: first path to compare
        -`path2`: second path to compare

    Best path processing will involve following steps:
    1.  Select a path with a reachable next hop.
    2.  Select the path with the highest weight.
    3.  If path weights are the same, select the path with the highest
        local preference value.
    4.  Prefer locally originated routes (network routes, redistributed
        routes, or aggregated routes) over received routes.
    5.  Select the route with the shortest AS-path length.
    6.  If all paths have the same AS-path length, select the path based
        on origin: IGP is preferred over EGP; EGP is preferred over
        Incomplete.
    7.  If the origins are the same, select the path with lowest MED
        value.
    8.  If the paths have the same MED values, select the path learned
        via EBGP over one learned via IBGP.
    9.  Select the route with the lowest IGP cost to the next hop.
    10. Select the route received from the peer with the lowest BGP
        router ID.
    11. Select the route received from the peer with the shorter
        CLUSTER_LIST length.

    Returns None if best-path among given paths cannot be computed else best
    path.
    Assumes paths from NC has source equal to None.
    """
    best_path = None
    best_path_reason = BPR_UNKNOWN
    if best_path is None:
        best_path = _cmp_by_reachable_nh(path1, path2)
        best_path_reason = BPR_REACHABLE_NEXT_HOP
    if best_path is None:
        best_path = _cmp_by_highest_wg(path1, path2)
        best_path_reason = BPR_HIGHEST_WEIGHT
    if best_path is None:
        best_path = _cmp_by_local_pref(path1, path2)
        best_path_reason = BPR_LOCAL_PREF
    if best_path is None:
        best_path = _cmp_by_local_origin(path1, path2)
        best_path_reason = BPR_LOCAL_ORIGIN
    if best_path is None:
        best_path = _cmp_by_aspath(path1, path2)
        best_path_reason = BPR_ASPATH
    if best_path is None:
        best_path = _cmp_by_origin(path1, path2)
        best_path_reason = BPR_ORIGIN
    if best_path is None:
        best_path = _cmp_by_med(path1, path2)
        best_path_reason = BPR_MED
    if best_path is None:
        best_path = _cmp_by_asn(local_asn, path1, path2)
        best_path_reason = BPR_ASN
    if best_path is None:
        best_path = _cmp_by_igp_cost(path1, path2)
        best_path_reason = BPR_IGP_COST
    if best_path is None:
        best_path = _cmp_by_router_id(local_asn, path1, path2)
        best_path_reason = BPR_ROUTER_ID
    if best_path is None:
        best_path = _cmp_by_cluster_list(path1, path2)
        best_path_reason = BPR_CLUSTER_LIST
    if best_path is None:
        best_path_reason = BPR_UNKNOWN
    return (best_path, best_path_reason)