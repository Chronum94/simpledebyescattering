"""
This file contains helper functions used by the XrayDebye class.

Also contains routine for calculation of atomic form factors and
X-ray wavelength dict.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial.distance import cdist, pdist

from simpledebyescattering.xrddata import read_waasmaier_coeffs


# Table (1) of
# D. WAASMAIER AND A. KIRFEL, Acta Cryst. (1995). A51, 416-431
# 10.1107/s0108767394013292

wavelengths = {
    "CuKa1": 1.5405981,
    "CuKa2": 1.54443,
    "CuKb1": 1.39225,
    "WLa1": 1.47642,
    "WLa2": 1.48748,
}


def initialize_atomic_form_factor_splines(
    symbols_list: list, s_vect: np.ndarray, data: str = "waasmaier"
) -> dict:
    atomic_form_factor_dict: dict = {}
    if data == "waasmaier":
        waasmaier_coefficients_dict = read_waasmaier_coeffs(symbols_list)
        for symbol, coefficients in waasmaier_coefficients_dict.items():
            atomic_form_factor_dict[symbol] = waasmaier_atomic_form_factor_spline(
                coefficients, s_vect
            )

    return atomic_form_factor_dict


def waasmaier_atomic_form_factor_spline(coeffs: list, s_vect: np.ndarray):
    ff = (
        np.sum(coeffs[:5] * np.exp(-s_vect[:, None] ** 2 * coeffs[-5:]), axis=1)
        + coeffs[5]
    )
    return InterpolatedUnivariateSpline(x=s_vect, y=ff, k=3)


def pdist_in_chunks(arr1, chunksize=5000):
    arr1len = arr1.shape[0]

    i = 0
    while (i * chunksize) < arr1len:
        j = i
        while (j * chunksize) < arr1len:
            if i == j:
                yield pdist(arr1[i * chunksize : (i + 1) * chunksize])
            else:
                yield cdist(
                    arr1[i * chunksize : (i + 1) * chunksize],
                    arr1[j * chunksize : (j + 1) * chunksize],
                ).ravel()

            j += 1
        i += 1


def cdist_in_chunks(arr1, arr2, chunksize=5000):
    arr1len = arr1.shape[0]
    arr2len = arr2.shape[0]

    i = 0
    while (i * chunksize) < arr1len:
        j = 0
        while (j * chunksize) < arr2len:
            yield cdist(
                arr1[i * chunksize : (i + 1) * chunksize],
                arr2[j * chunksize : (j + 1) * chunksize],
            ).ravel()
            j += 1
        i += 1
