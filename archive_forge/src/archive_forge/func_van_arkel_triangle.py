from __future__ import annotations
import importlib
import math
from functools import wraps
from string import ascii_letters
from typing import TYPE_CHECKING, Literal
import matplotlib.pyplot as plt
import numpy as np
import palettable.colorbrewer.diverging
from matplotlib import cm, colors
from pymatgen.core import Element
def van_arkel_triangle(list_of_materials: Sequence, annotate: bool=True):
    """A static method that generates a binary van Arkel-Ketelaar triangle to
    quantify the ionic, metallic and covalent character of a compound
    by plotting the electronegativity difference (y) vs average (x).
    See:
        A.E. van Arkel, Molecules and Crystals in Inorganic Chemistry,
            Interscience, New York (1956)
    and
        J.A.A Ketelaar, Chemical Constitution (2nd edition), An Introduction
            to the Theory of the Chemical Bond, Elsevier, New York (1958).

    Args:
        list_of_materials (list): A list of computed entries of binary
            materials or a list of lists containing two elements (str).
        annotate (bool): Whether or not to label the points on the
            triangle with reduced formula (if list of entries) or pair
            of elements (if list of list of str).

    Returns:
        plt.Axes: matplotlib Axes object
    """
    pt1 = np.array([(Element('F').X + Element('Fr').X) / 2, abs(Element('F').X - Element('Fr').X)])
    pt2 = np.array([(Element('Cs').X + Element('Fr').X) / 2, abs(Element('Cs').X - Element('Fr').X)])
    pt3 = np.array([(Element('O').X + Element('F').X) / 2, abs(Element('O').X - Element('F').X)])
    d = np.array(pt1) - np.array(pt2)
    slope1 = d[1] / d[0]
    b1 = pt1[1] - slope1 * pt1[0]
    d = pt3 - pt1
    slope2 = d[1] / d[0]
    b2 = pt3[1] - slope2 * pt3[0]
    plt.xlim(pt2[0] - 0.45, -b2 / slope2 + 0.45)
    plt.ylim(-0.45, pt1[1] + 0.45)
    plt.annotate('Ionic', xy=[pt1[0] - 0.3, pt1[1] + 0.05], fontsize=20)
    plt.annotate('Covalent', xy=[-b2 / slope2 - 0.65, -0.4], fontsize=20)
    plt.annotate('Metallic', xy=[pt2[0] - 0.4, -0.4], fontsize=20)
    plt.xlabel('$\\frac{\\chi_{A}+\\chi_{B}}{2}$', fontsize=25)
    plt.ylabel('$|\\chi_{A}-\\chi_{B}|$', fontsize=25)
    chi_list = [el.X for el in Element]
    plt.plot([min(chi_list), pt1[0]], [slope1 * min(chi_list) + b1, pt1[1]], 'k-', linewidth=3)
    plt.plot([pt1[0], -b2 / slope2], [pt1[1], 0], 'k-', linewidth=3)
    plt.plot([min(chi_list), -b2 / slope2], [0, 0], 'k-', linewidth=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.gca()
    ax.fill_between([min(chi_list), pt1[0]], [slope1 * min(chi_list) + b1, pt1[1]], facecolor=[1, 1, 0], zorder=-5, edgecolor=[1, 1, 0])
    ax.fill_between([pt1[0], -b2 / slope2], [pt1[1], slope2 * min(chi_list) - b1], facecolor=[1, 1, 0], zorder=-5, edgecolor=[1, 1, 0])
    x_pt = Element('Pt').X
    ax.fill_between([min(chi_list), (x_pt + min(chi_list)) / 2], [0, slope1 * (x_pt + min(chi_list)) / 2 + b1], facecolor=[1, 0, 0], zorder=-3, alpha=0.8)
    ax.fill_between([(x_pt + min(chi_list)) / 2, x_pt], [slope1 * ((x_pt + min(chi_list)) / 2) + b1, 0], facecolor=[1, 0, 0], zorder=-3, alpha=0.8)
    ax.fill_between([(x_pt + min(chi_list)) / 2, ((x_pt + min(chi_list)) / 2 + -b2 / slope2) / 2], [0, slope2 * (((x_pt + min(chi_list)) / 2 + -b2 / slope2) / 2) + b2], facecolor=[0, 1, 0], zorder=-4, alpha=0.8)
    ax.fill_between([((x_pt + min(chi_list)) / 2 + -b2 / slope2) / 2, -b2 / slope2], [slope2 * (((x_pt + min(chi_list)) / 2 + -b2 / slope2) / 2) + b2, 0], facecolor=[0, 1, 0], zorder=-4, alpha=0.8)
    for entry in list_of_materials:
        if type(entry).__name__ not in ['ComputedEntry', 'ComputedStructureEntry']:
            X_pair = [Element(el).X for el in entry]
            el_1, el_2 = entry
            formatted_formula = f'{el_1}-{el_2}'
        else:
            X_pair = [Element(el).X for el in entry.composition.as_dict()]
            formatted_formula = format_formula(entry.reduced_formula)
        plt.scatter(np.mean(X_pair), abs(X_pair[0] - X_pair[1]), c='b', s=100)
        if annotate:
            plt.annotate(formatted_formula, fontsize=15, xy=[np.mean(X_pair) + 0.005, abs(X_pair[0] - X_pair[1])])
    plt.tight_layout()
    return ax