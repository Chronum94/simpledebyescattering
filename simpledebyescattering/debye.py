from itertools import combinations

import numpy as np

from simpledebyescattering.xrayfunctions import (
    initialize_atomic_form_factor_splines,
    pdist_in_chunks,
    cdist_in_chunks,
)

from multimethod import multimethod


def one_species_contribution(positions, form_factor_spline, s) -> np.ndarray:

    contribution = np.zeros_like(s)
    n_atoms = positions.shape[0]

    for distances in pdist_in_chunks(positions):
        for i in range(s.shape[0]):
            contribution[i] += form_factor_spline(s[i]) ** 2 * (
                n_atoms + 2 * np.sum(np.sinc(distances * s[i] * 2))
            )

    return contribution


def one_species_hist_contribution(
    positions, form_factor_spline, s, bin_width=1e-3
) -> np.ndarray:
    n_atoms: int = positions.shape[0]
    contribution: np.ndarray = np.zeros_like(s)

    if positions.shape[0] == 1:
        return form_factor_spline(s) ** 2 * n_atoms

    for distances in pdist_in_chunks(positions):
        nbins = int(np.ceil(np.ptp(distances) / bin_width)) + 1

        dist_hist, bin_edges = np.histogram(distances, bins=nbins)
        nontrivial_bins = np.nonzero(dist_hist)

        dist_hist, bin_edges = (
            dist_hist[nontrivial_bins],
            bin_edges[nontrivial_bins] + (bin_edges[1] - bin_edges[0]) / 2,
        )

        for i in range(s.shape[0]):
            contribution[i] += form_factor_spline(s[i]) ** 2 * (
                n_atoms + 2 * np.sum(dist_hist * np.sinc(bin_edges * s[i] * 2))
            )

    return contribution


def two_species_contribution(
    positions1, positions2, form_factor_spline1, form_factor_spline2, s
) -> np.ndarray:
    contribution = np.zeros_like(s)

    for distances in cdist_in_chunks(positions1, positions2):
        for i in range(s.shape[0]):
            contribution[i] += (
                2.0
                * form_factor_spline1(s[i])
                * form_factor_spline2(s[i])
                * np.sum(np.sinc(distances * s[i] * 2))
            )

    return contribution


def two_species_hist_contribution(
    positions1, positions2, form_factor_spline1, form_factor_spline2, s, bin_width=1e-2
) -> np.ndarray:
    contribution = np.zeros_like(s)

    for distances in cdist_in_chunks(positions1, positions2):
        nbins = int(np.ceil(np.ptp(distances) / bin_width)) + 1

        dist_hist, bin_edges = np.histogram(distances, bins=nbins)
        nontrivial_bins = np.nonzero(dist_hist)
        dist_hist, bin_edges = (
            dist_hist[nontrivial_bins],
            bin_edges[nontrivial_bins] + (bin_edges[1] - bin_edges[0]) / 2,
        )
        for i in range(s.shape[0]):
            contribution[i] += (
                2.0
                * form_factor_spline1(s[i])
                * form_factor_spline2(s[i])
                * np.sum(dist_hist * np.sinc(bin_edges * s[i] * 2))
            )

    return contribution

# class XRDData:
#     mode = "XRD"
#     xvalues_name = "2theta"

#     def __init__(self, twotheta, intensities):
#         self.twotheta = twotheta
#         self.intensities = intensities

#     @classmethod
#     def calculate(cls, xrd, twotheta):
#         if twotheta is None:
#             twotheta = np.linspace(15, 55, 100)
#         else:
#             twotheta = np.asarray(twotheta)

#         svalues = 2 * np.sin(twotheta * np.pi / 180 / 2.0) / xrd.wavelength
#         result = xrd.get(svalues)
#         return cls(twotheta, np.array(result))

#     def xvalues(self):
#         return self.twotheta.copy()


# class SAXSData:
#     mode = "SAXS"
#     xvalues_name = "q(1/Å)"

#     def __init__(self, qvalues, intensities):
#         self.qvalues = qvalues
#         self.intensities = intensities

#     @classmethod
#     def calculate(cls, xrd, qvalues):
#         if qvalues is None:
#             qvalues = np.logspace(-3, -0.3, 100)
#         else:
#             qvalues = np.asarray(qvalues)

#         svalues = qvalues / (2 * np.pi)
#         result = xrd.get(svalues)
#         return cls(qvalues, result)

#     def xvalues(self):
#         return self.qvalues.copy()


# output_data_class = {"XRD": XRDData, "SAXS": SAXSData}


class XrayDebye:
    """
    Class for calculation of XRD or SAXS patterns.
    """

    def __init__(
        self,
        atoms: "ase.Atoms",
        wavelength: float,
        damping: float = 0.04,
        method: str = "Iwasa",
        alpha: float = 1.01,
        histogram_approximation: bool = True,
    ):
        """[summary]

        Parameters
        ----------
        atoms : ase.Atoms
            atoms object for which calculation will be performed
        wavelength : float
            X-ray wavelength in Angstrom. Used for XRD and to setup dumpings
        damping : float, optional
            thermal damping factor parameter (B-factor), by default 0.04
        method : str, optional
            method of calculation (damping and atomic factors affected), by default "Iwasa"

            If set to 'Iwasa' than angular damping and q-dependence of
            atomic factors are used.

            For any other string there will be only thermal damping
            and constant atomic factors (`f_a(q) = Z_a`)
        alpha : float, optional
            parameter for angular damping of scattering intensity.
            Close to 1.0 for unplorized beam, by default 1.01
        histogram_approximation : bool, optional
            [description], by default True
        """

        self.wavelength = wavelength
        self.method = method
        self.alpha = alpha

        self.damping = damping
        self.atoms = atoms

        self.atomic_form_factor_dict = initialize_atomic_form_factor_splines(
            set(self.atoms.symbols), np.linspace(0, 6, 500)
        )

        self.hist_approx = histogram_approximation

    def set_damping(self, damping):
        """set B-factor for thermal damping"""
        self.damping = damping

from typing import Iterable
@multimethod
def calc_pattern(xrdobj:XrayDebye, x:Iterable[float], mode:str):
    r"""
    Calculate X-ray diffraction pattern or
    small angle X-ray scattering pattern.

    Parameters:

    x: float array
        points where intensity will be calculated.
        XRD - 2theta values, in degrees;
        SAXS - q values in 1/Å
        (`q = 2 \pi \cdot s = 4 \pi \sin( \theta) / \lambda`).
        If ``x`` is ``None`` then default values will be used.

    mode: {'XRD', 'SAXS'}
        the mode of calculation: X-ray diffraction (XRD) or
        small-angle scattering (SAXS).

    Returns:
        list of intensities calculated for values given in ``x``.
    """
    assert mode in ["XRD", "SAXS"]
    x = np.array(x)
    if mode == "XRD":
        svalues = 2 * np.sin(x * np.pi / 180 / 2.0) / xrdobj.wavelength
    elif mode == "SAXS":
        svalues = x / (2 * np.pi)


    # cls = output_data_class["XRD"]

    data = _calculate(xrdobj, svalues)
    return data.intensities

@multimethod
def _calculate(xrdobj: XrayDebye, s: np.ndarray):
    r"""Get the powder x-ray (XRD) scattering intensity
    using the Debye-Formula at single point.

    Parameters:

    s: float array, in inverse Angstrom
        scattering vector value (`s = q / 2\pi`).

    Returns:
        Intensity at given scattering vector `s`.
    """

    pre = np.exp(-xrdobj.damping * s**2 / 2)

    if xrdobj.method == "Iwasa":
        sinth = xrdobj.wavelength * s / 2.0
        positive = 1.0 - sinth**2
        positive[positive < 0] = 0
        costh = np.sqrt(positive)
        cos2th = np.cos(2.0 * np.arccos(costh))
        pre *= costh / (1.0 + xrdobj.alpha * cos2th**2)

    I = np.zeros_like(s)

    # Calculate contribution from pairs of same atomic species
    symbols = set(xrdobj.atoms.symbols)

    for symbol in symbols:
        # print(f"Calculating on-diagonal {symbol} block")
        if not xrdobj.hist_approx:
            I[:] += one_species_contribution(
                xrdobj.atoms[xrdobj.atoms.symbols == symbol].positions,
                xrdobj.atomic_form_factor_dict[symbol],
                s,
            )
        else:
            I[:] += one_species_hist_contribution(
                xrdobj.atoms[xrdobj.atoms.symbols == symbol].positions,
                xrdobj.atomic_form_factor_dict[symbol],
                s,
            )

    # Calculation contribution from pairs of different atomic species
    symbols_pairs = combinations(symbols, 2)

    for symbols_pair in symbols_pairs:
        symbol1, symbol2 = symbols_pair
        # print(f"Calculating off-diagonal {symbol1}+{symbol2} blocks")
        if not xrdobj.hist_approx:
            I[:] += two_species_contribution(
                xrdobj.atoms[xrdobj.atoms.symbols == symbol1].positions,
                xrdobj.atoms[xrdobj.atoms.symbols == symbol2].positions,
                xrdobj.atomic_form_factor_dict[symbol1],
                xrdobj.atomic_form_factor_dict[symbol2],
                s,
            )

        else:
            I[:] += two_species_hist_contribution(
                xrdobj.atoms[xrdobj.atoms.symbols == symbol1].positions,
                xrdobj.atoms[xrdobj.atoms.symbols == symbol2].positions,
                xrdobj.atomic_form_factor_dict[symbol1],
                xrdobj.atomic_form_factor_dict[symbol2],
                s,
            )

    # lin_zhigilei_factor = [len(xrdobj.atoms.symbols == x) * xrdobj.atomic_form_factor_dict[x](s) ** 2 for x in symbols]

    return pre * I  # / np.sum(lin_zhigilei_factor, axis=0)