from __future__ import annotations
import logging
import warnings
from fractions import Fraction
from functools import reduce
from itertools import chain, combinations, product
from math import cos, floor, gcd
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.fractions import lcm
from numpy.testing import assert_allclose
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def get_trans_mat(r_axis, angle, normal=False, trans_cry=None, lat_type='c', ratio=None, surface=None, max_search=20, quick_gen=False):
    """
        Find the two transformation matrix for each grain from given rotation axis,
        GB plane, rotation angle and corresponding ratio (see explanation for ratio
        below).
        The structure of each grain can be obtained by applying the corresponding
        transformation matrix to the conventional cell.
        The algorithm for this code is from reference, Acta Cryst, A32,783(1976).

        Args:
            r_axis (list of 3 integers, e.g. u, v, w or 4 integers, e.g. u, v, t, w for hex/rho system only): the
                rotation axis of the grain boundary.
            angle (float, in unit of degree): the rotation angle of the grain boundary
            normal (logic): determine if need to require the c axis of one grain associated with
                the first transformation matrix perpendicular to the surface or not.
                default to false.
            trans_cry (np.array): shape 3x3. If the structure given are primitive cell in cubic system, e.g.
                bcc or fcc system, trans_cry is the transformation matrix from its
                conventional cell to the primitive cell.
            lat_type (str): one character to specify the lattice type. Defaults to 'c' for cubic.
                'c' or 'C': cubic system
                't' or 'T': tetragonal system
                'o' or 'O': orthorhombic system
                'h' or 'H': hexagonal system
                'r' or 'R': rhombohedral system
            ratio (list of integers): lattice axial ratio.
                For cubic system, ratio is not needed.
                For tetragonal system, ratio = [mu, mv], list of two integers, that is, mu/mv = c2/a2. If it is
                irrational, set it to none.
                For orthorhombic system, ratio = [mu, lam, mv], list of 3 integers, that is, mu:lam:mv = c2:b2:a2.
                If irrational for one axis, set it to None. e.g. mu:lam:mv = c2,None,a2, means b2 is irrational.
                For rhombohedral system, ratio = [mu, mv], list of two integers,
                that is, mu/mv is the ratio of (1+2*cos(alpha)/cos(alpha).
                If irrational, set it to None.
                For hexagonal system, ratio = [mu, mv], list of two integers,
                that is, mu/mv = c2/a2. If it is irrational, set it to none.
            surface (list of 3 integers, e.g. h, k, l or 4 integers, e.g. h, k, i, l for hex/rho system only): The
                miller index of grain boundary plane, with the format of [h,k,l] if surface is not given, the default
                is perpendicular to r_axis, which is a twist grain boundary.
            max_search (int): max search for the GB lattice vectors that give the smallest GB
                lattice. If normal is true, also max search the GB c vector that perpendicular
                to the plane.
            quick_gen (bool): whether to quickly generate a supercell, if set to true, no need to
                find the smallest cell.

        Returns:
            t1 (3 by 3 integer array): The transformation array for one grain.
            t2 (3 by 3 integer array): The transformation array for the other grain
        """
    if trans_cry is None:
        trans_cry = np.eye(3)
    if len(r_axis) == 4:
        u1 = r_axis[0]
        v1 = r_axis[1]
        w1 = r_axis[3]
        if lat_type.lower() == 'h':
            u = 2 * u1 + v1
            v = 2 * v1 + u1
            w = w1
            r_axis = [u, v, w]
        elif lat_type.lower() == 'r':
            u = 2 * u1 + v1 + w1
            v = v1 + w1 - u1
            w = w1 - 2 * v1 - u1
            r_axis = [u, v, w]
    if reduce(gcd, r_axis) != 1:
        r_axis = [int(round(x / reduce(gcd, r_axis))) for x in r_axis]
    if surface is not None and len(surface) == 4:
        u1 = surface[0]
        v1 = surface[1]
        w1 = surface[3]
        surface = [u1, v1, w1]
    if surface is None:
        if lat_type.lower() == 'c':
            surface = r_axis
        else:
            if lat_type.lower() == 'h':
                c2_a2_ratio = 1.0 if ratio is None else ratio[0] / ratio[1]
                metric = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, c2_a2_ratio]])
            elif lat_type.lower() == 'r':
                cos_alpha = 0.5 if ratio is None else 1.0 / (ratio[0] / ratio[1] - 2)
                metric = np.array([[1, cos_alpha, cos_alpha], [cos_alpha, 1, cos_alpha], [cos_alpha, cos_alpha, 1]])
            elif lat_type.lower() == 't':
                c2_a2_ratio = 1.0 if ratio is None else ratio[0] / ratio[1]
                metric = np.array([[1, 0, 0], [0, 1, 0], [0, 0, c2_a2_ratio]])
            elif lat_type.lower() == 'o':
                for idx in range(3):
                    if ratio[idx] is None:
                        ratio[idx] = 1
                metric = np.array([[1, 0, 0], [0, ratio[1] / ratio[2], 0], [0, 0, ratio[0] / ratio[2]]])
            else:
                raise RuntimeError('Lattice type has not implemented.')
            surface = np.matmul(r_axis, metric)
            fractions = [Fraction(x).limit_denominator() for x in surface]
            least_mul = reduce(lcm, [fraction.denominator for fraction in fractions])
            surface = [int(round(x * least_mul)) for x in surface]
    if reduce(gcd, surface) != 1:
        index = reduce(gcd, surface)
        surface = [int(round(x / index)) for x in surface]
    if lat_type.lower() == 'h':
        u, v, w = r_axis
        if ratio is None:
            mu, mv = [1, 1]
            if w != 0 and (u != 0 or v != 0):
                raise RuntimeError('For irrational c2/a2, CSL only exist for [0,0,1] or [u,v,0] and m = 0')
        else:
            mu, mv = ratio
        if gcd(mu, mv) != 1:
            temp = gcd(mu, mv)
            mu = int(round(mu / temp))
            mv = int(round(mv / temp))
        d = (u ** 2 + v ** 2 - u * v) * mv + w ** 2 * mu
        if abs(angle - 180.0) < 1.0:
            m = 0
            n = 1
        else:
            fraction = Fraction(np.tan(angle / 2 / 180.0 * np.pi) / np.sqrt(float(d) / 3.0 / mu)).limit_denominator()
            m = fraction.denominator
            n = fraction.numerator
        r_list = [(u ** 2 * mv - v ** 2 * mv - w ** 2 * mu) * n ** 2 + 2 * w * mu * m * n + 3 * mu * m ** 2, (2 * v - u) * u * mv * n ** 2 - 4 * w * mu * m * n, 2 * u * w * mu * n ** 2 + 2 * (2 * v - u) * mu * m * n, (2 * u - v) * v * mv * n ** 2 + 4 * w * mu * m * n, (v ** 2 * mv - u ** 2 * mv - w ** 2 * mu) * n ** 2 - 2 * w * mu * m * n + 3 * mu * m ** 2, 2 * v * w * mu * n ** 2 - 2 * (2 * u - v) * mu * m * n, (2 * u - v) * w * mv * n ** 2 - 3 * v * mv * m * n, (2 * v - u) * w * mv * n ** 2 + 3 * u * mv * m * n, (w ** 2 * mu - u ** 2 * mv - v ** 2 * mv + u * v * mv) * n ** 2 + 3 * mu * m ** 2]
        m = -1 * m
        r_list_inv = [(u ** 2 * mv - v ** 2 * mv - w ** 2 * mu) * n ** 2 + 2 * w * mu * m * n + 3 * mu * m ** 2, (2 * v - u) * u * mv * n ** 2 - 4 * w * mu * m * n, 2 * u * w * mu * n ** 2 + 2 * (2 * v - u) * mu * m * n, (2 * u - v) * v * mv * n ** 2 + 4 * w * mu * m * n, (v ** 2 * mv - u ** 2 * mv - w ** 2 * mu) * n ** 2 - 2 * w * mu * m * n + 3 * mu * m ** 2, 2 * v * w * mu * n ** 2 - 2 * (2 * u - v) * mu * m * n, (2 * u - v) * w * mv * n ** 2 - 3 * v * mv * m * n, (2 * v - u) * w * mv * n ** 2 + 3 * u * mv * m * n, (w ** 2 * mu - u ** 2 * mv - v ** 2 * mv + u * v * mv) * n ** 2 + 3 * mu * m ** 2]
        m = -1 * m
        F = 3 * mu * m ** 2 + d * n ** 2
        all_list = r_list + r_list_inv + [F]
        com_fac = reduce(gcd, all_list)
        sigma = F / com_fac
        r_matrix = (np.array(r_list) / com_fac / sigma).reshape(3, 3)
    elif lat_type.lower() == 'r':
        u, v, w = r_axis
        if ratio is None:
            mu, mv = [1, 1]
            if u + v + w != 0 and (u != v or u != w):
                raise RuntimeError('For irrational ratio_alpha, CSL only exist for [1,1,1] or [u, v, -(u+v)] and m =0')
        else:
            mu, mv = ratio
        if gcd(mu, mv) != 1:
            temp = gcd(mu, mv)
            mu = int(round(mu / temp))
            mv = int(round(mv / temp))
        d = (u ** 2 + v ** 2 + w ** 2) * (mu - 2 * mv) + 2 * mv * (v * w + w * u + u * v)
        if abs(angle - 180.0) < 1.0:
            m = 0
            n = 1
        else:
            fraction = Fraction(np.tan(angle / 2 / 180.0 * np.pi) / np.sqrt(float(d) / mu)).limit_denominator()
            m = fraction.denominator
            n = fraction.numerator
        r_list = [(mu - 2 * mv) * (u ** 2 - v ** 2 - w ** 2) * n ** 2 + 2 * mv * (v - w) * m * n - 2 * mv * v * w * n ** 2 + mu * m ** 2, 2 * (mv * u * n * (w * n + u * n - m) - (mu - mv) * m * w * n + (mu - 2 * mv) * u * v * n ** 2), 2 * (mv * u * n * (v * n + u * n + m) + (mu - mv) * m * v * n + (mu - 2 * mv) * w * u * n ** 2), 2 * (mv * v * n * (w * n + v * n + m) + (mu - mv) * m * w * n + (mu - 2 * mv) * u * v * n ** 2), (mu - 2 * mv) * (v ** 2 - w ** 2 - u ** 2) * n ** 2 + 2 * mv * (w - u) * m * n - 2 * mv * u * w * n ** 2 + mu * m ** 2, 2 * (mv * v * n * (v * n + u * n - m) - (mu - mv) * m * u * n + (mu - 2 * mv) * w * v * n ** 2), 2 * (mv * w * n * (w * n + v * n - m) - (mu - mv) * m * v * n + (mu - 2 * mv) * w * u * n ** 2), 2 * (mv * w * n * (w * n + u * n + m) + (mu - mv) * m * u * n + (mu - 2 * mv) * w * v * n ** 2), (mu - 2 * mv) * (w ** 2 - u ** 2 - v ** 2) * n ** 2 + 2 * mv * (u - v) * m * n - 2 * mv * u * v * n ** 2 + mu * m ** 2]
        m = -1 * m
        r_list_inv = [(mu - 2 * mv) * (u ** 2 - v ** 2 - w ** 2) * n ** 2 + 2 * mv * (v - w) * m * n - 2 * mv * v * w * n ** 2 + mu * m ** 2, 2 * (mv * u * n * (w * n + u * n - m) - (mu - mv) * m * w * n + (mu - 2 * mv) * u * v * n ** 2), 2 * (mv * u * n * (v * n + u * n + m) + (mu - mv) * m * v * n + (mu - 2 * mv) * w * u * n ** 2), 2 * (mv * v * n * (w * n + v * n + m) + (mu - mv) * m * w * n + (mu - 2 * mv) * u * v * n ** 2), (mu - 2 * mv) * (v ** 2 - w ** 2 - u ** 2) * n ** 2 + 2 * mv * (w - u) * m * n - 2 * mv * u * w * n ** 2 + mu * m ** 2, 2 * (mv * v * n * (v * n + u * n - m) - (mu - mv) * m * u * n + (mu - 2 * mv) * w * v * n ** 2), 2 * (mv * w * n * (w * n + v * n - m) - (mu - mv) * m * v * n + (mu - 2 * mv) * w * u * n ** 2), 2 * (mv * w * n * (w * n + u * n + m) + (mu - mv) * m * u * n + (mu - 2 * mv) * w * v * n ** 2), (mu - 2 * mv) * (w ** 2 - u ** 2 - v ** 2) * n ** 2 + 2 * mv * (u - v) * m * n - 2 * mv * u * v * n ** 2 + mu * m ** 2]
        m = -1 * m
        F = mu * m ** 2 + d * n ** 2
        all_list = r_list_inv + r_list + [F]
        com_fac = reduce(gcd, all_list)
        sigma = F / com_fac
        r_matrix = (np.array(r_list) / com_fac / sigma).reshape(3, 3)
    else:
        u, v, w = r_axis
        if lat_type.lower() == 'c':
            mu = 1
            lam = 1
            mv = 1
        elif lat_type.lower() == 't':
            if ratio is None:
                mu, mv = [1, 1]
                if w != 0 and (u != 0 or v != 0):
                    raise RuntimeError('For irrational c2/a2, CSL only exist for [0,0,1] or [u,v,0] and m = 0')
            else:
                mu, mv = ratio
            lam = mv
        elif lat_type.lower() == 'o':
            if None in ratio:
                mu, lam, mv = ratio
                non_none = [i for i in ratio if i is not None]
                if len(non_none) < 2:
                    raise RuntimeError('No CSL exist for two irrational numbers')
                non1, non2 = non_none
                if mu is None:
                    lam = non1
                    mv = non2
                    mu = 1
                    if w != 0 and (u != 0 or v != 0):
                        raise RuntimeError('For irrational c2, CSL only exist for [0,0,1] or [u,v,0] and m = 0')
                elif lam is None:
                    mu = non1
                    mv = non2
                    lam = 1
                    if v != 0 and (u != 0 or w != 0):
                        raise RuntimeError('For irrational b2, CSL only exist for [0,1,0] or [u,0,w] and m = 0')
                elif mv is None:
                    mu = non1
                    lam = non2
                    mv = 1
                    if u != 0 and (w != 0 or v != 0):
                        raise RuntimeError('For irrational a2, CSL only exist for [1,0,0] or [0,v,w] and m = 0')
            else:
                mu, lam, mv = ratio
                if u == 0 and v == 0:
                    mu = 1
                if u == 0 and w == 0:
                    lam = 1
                if v == 0 and w == 0:
                    mv = 1
        if reduce(gcd, [mu, lam, mv]) != 1:
            temp = reduce(gcd, [mu, lam, mv])
            mu = int(round(mu / temp))
            mv = int(round(mv / temp))
            lam = int(round(lam / temp))
        d = (mv * u ** 2 + lam * v ** 2) * mv + w ** 2 * mu * mv
        if abs(angle - 180.0) < 1.0:
            m = 0
            n = 1
        else:
            fraction = Fraction(np.tan(angle / 2 / 180.0 * np.pi) / np.sqrt(d / mu / lam)).limit_denominator()
            m = fraction.denominator
            n = fraction.numerator
        r_list = [(u ** 2 * mv * mv - lam * v ** 2 * mv - w ** 2 * mu * mv) * n ** 2 + lam * mu * m ** 2, 2 * lam * (v * u * mv * n ** 2 - w * mu * m * n), 2 * mu * (u * w * mv * n ** 2 + v * lam * m * n), 2 * mv * (u * v * mv * n ** 2 + w * mu * m * n), (v ** 2 * mv * lam - u ** 2 * mv * mv - w ** 2 * mu * mv) * n ** 2 + lam * mu * m ** 2, 2 * mv * mu * (v * w * n ** 2 - u * m * n), 2 * mv * (u * w * mv * n ** 2 - v * lam * m * n), 2 * lam * mv * (v * w * n ** 2 + u * m * n), (w ** 2 * mu * mv - u ** 2 * mv * mv - v ** 2 * mv * lam) * n ** 2 + lam * mu * m ** 2]
        m = -1 * m
        r_list_inv = [(u ** 2 * mv * mv - lam * v ** 2 * mv - w ** 2 * mu * mv) * n ** 2 + lam * mu * m ** 2, 2 * lam * (v * u * mv * n ** 2 - w * mu * m * n), 2 * mu * (u * w * mv * n ** 2 + v * lam * m * n), 2 * mv * (u * v * mv * n ** 2 + w * mu * m * n), (v ** 2 * mv * lam - u ** 2 * mv * mv - w ** 2 * mu * mv) * n ** 2 + lam * mu * m ** 2, 2 * mv * mu * (v * w * n ** 2 - u * m * n), 2 * mv * (u * w * mv * n ** 2 - v * lam * m * n), 2 * lam * mv * (v * w * n ** 2 + u * m * n), (w ** 2 * mu * mv - u ** 2 * mv * mv - v ** 2 * mv * lam) * n ** 2 + lam * mu * m ** 2]
        m = -1 * m
        F = mu * lam * m ** 2 + d * n ** 2
        all_list = r_list + r_list_inv + [F]
        com_fac = reduce(gcd, all_list)
        sigma = F / com_fac
        r_matrix = (np.array(r_list) / com_fac / sigma).reshape(3, 3)
    if sigma > 1000:
        raise RuntimeError('Sigma >1000 too large. Are you sure what you are doing, Please check the GB if exist')
    surface = np.matmul(surface, np.transpose(trans_cry))
    fractions = [Fraction(x).limit_denominator() for x in surface]
    least_mul = reduce(lcm, [fraction.denominator for fraction in fractions])
    surface = [int(round(x * least_mul)) for x in surface]
    if reduce(gcd, surface) != 1:
        index = reduce(gcd, surface)
        surface = [int(round(x / index)) for x in surface]
    r_axis = np.rint(np.matmul(r_axis, np.linalg.inv(trans_cry))).astype(int)
    if reduce(gcd, r_axis) != 1:
        r_axis = [int(round(x / reduce(gcd, r_axis))) for x in r_axis]
    r_matrix = np.dot(np.dot(np.linalg.inv(trans_cry.T), r_matrix), trans_cry.T)
    eye = np.eye(3, dtype=int)
    for hh in range(3):
        if abs(r_axis[hh]) != 0:
            eye[hh] = np.array(r_axis)
            kk = hh + 1 if hh + 1 < 3 else abs(2 - hh)
            ll = hh + 2 if hh + 2 < 3 else abs(1 - hh)
            break
    trans = eye.T
    new_rot = np.array(r_matrix)
    fractions = [Fraction(x).limit_denominator() for x in new_rot[:, kk]]
    least_mul = reduce(lcm, [fraction.denominator for fraction in fractions])
    scale = np.zeros((3, 3))
    scale[hh, hh] = 1
    scale[kk, kk] = least_mul
    scale[ll, ll] = sigma / least_mul
    for idx in range(least_mul):
        check_int = idx * new_rot[:, kk] + sigma / least_mul * new_rot[:, ll]
        if all((np.round(x, 5).is_integer() for x in list(check_int))):
            n_final = idx
            break
    try:
        n_final
    except NameError:
        raise RuntimeError('Something is wrong. Check if this GB exists or not')
    scale[kk, ll] = n_final
    csl_init = np.rint(np.dot(np.dot(r_matrix, trans), scale)).astype(int).T
    if abs(r_axis[hh]) > 1:
        csl_init = GrainBoundaryGenerator.reduce_mat(np.array(csl_init), r_axis[hh], r_matrix)
    csl = np.rint(Lattice(csl_init).get_niggli_reduced_lattice().matrix).astype(int)
    if lat_type.lower() != 'c':
        if lat_type.lower() == 'h':
            trans_cry = np.array([[1, 0, 0], [-0.5, np.sqrt(3.0) / 2.0, 0], [0, 0, np.sqrt(mu / mv)]])
        elif lat_type.lower() == 'r':
            c2_a2_ratio = 1.0 if ratio is None else 3.0 / (2 - 6 * mv / mu)
            trans_cry = np.array([[0.5, np.sqrt(3.0) / 6.0, 1.0 / 3 * np.sqrt(c2_a2_ratio)], [-0.5, np.sqrt(3.0) / 6.0, 1.0 / 3 * np.sqrt(c2_a2_ratio)], [0, -1 * np.sqrt(3.0) / 3.0, 1.0 / 3 * np.sqrt(c2_a2_ratio)]])
        else:
            trans_cry = np.array([[1, 0, 0], [0, np.sqrt(lam / mv), 0], [0, 0, np.sqrt(mu / mv)]])
    t1_final = GrainBoundaryGenerator.slab_from_csl(csl, surface, normal, trans_cry, max_search=max_search, quick_gen=quick_gen)
    t2_final = np.array(np.rint(np.dot(t1_final, np.linalg.inv(r_matrix.T)))).astype(int)
    return (t1_final, t2_final)