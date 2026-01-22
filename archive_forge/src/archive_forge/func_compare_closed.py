import snappy
import regina
import snappy.snap.t3mlite as t3m
import snappy.snap.t3mlite.spun as spun
def compare_closed(snappy_manifold):
    N = snappy_manifold.filled_triangulation()
    T = t3m.Mcomplex(N)
    T.find_normal_surfaces()
    t_hashes = sorted((hash_t3m_surface(S) for S in T.NormalSurfaces))
    R = to_regina(N)
    r_hashes = sorted((hash_regina_surface(S) for S in vertex_surfaces(R)))
    all_together = sum(t_hashes, [])
    return (t_hashes == r_hashes, len(all_together), sum(all_together))