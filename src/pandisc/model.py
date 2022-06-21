"""
Functions to compute the pandisc model. For the formulation please refer to
the documentation.
"""

import numpy as np


def disk(v_r, k, v_sigma, f, v_c, v_arr):
    """
    compute the co-rotating disk part of the model, the integrated model flux is
    normalized to f

    :param float v_r: float, rotation velocity of the disk
    :param float k: float, gradient of the angular distribution, defining the
        asymmetry of the line, should be in the range (−2/𝜋,2/𝜋)
    :param float v_sigma: float, sigma of the disk velocity dispersion
    :param float f: float, the integrated flux integrate(f_model dv) of the model
    :param float v_c: float, velocity of the line center
    :param v_arr: array, recording the axis to compute the model
    :type v_arr: numpy.ndarray or float
    :return: array of the computed model, in the same shape as input v_arr
    :rtype: numpy.ndarray or float
    """

    line_disk = np.zeros(v_arr.size)
    flag = (v_c - v_r - 5 * v_sigma < v_arr) & (v_arr < v_c + v_r + 5 * v_sigma)
    # only compute the channels expected to have flux

    n_sample = min(int(v_r / v_sigma) * 2, 200)
    dphi = np.pi / max(n_sample, 20)
    phiphi = np.arange(-np.pi / 2 + dphi / 2, np.pi / 2, dphi).reshape(-1, 1)
    sinphi = np.sin(phiphi)
    vv_los = v_arr[flag].reshape(1, -1)

    line_disk[flag] = ((1 + k * phiphi) *
                       np.exp(-(v_r * sinphi + (vv_los - v_c)) ** 2 /
                              (2 * v_sigma ** 2))).sum(axis=0) \
                      * dphi / np.sqrt(2 * np.pi) / np.pi / v_sigma * f

    return line_disk


def gaussian(v_g, f, v_c, v_arr):
    """
    compute the gaussian component of the model, the integrated flux is
    normalized to f

    :param float v_g: float, sigma of the gaussian peak
    :param float f: float, the integrated flux integrate(f_model dv) of the model
    :param float v_c: float, velocity of the line center
    :param v_arr: array, recording the axis to compute the model
    :type v_arr: numpy.ndarray or float
    :return: array of the computed model, in the same shape as input v_arr
    :rtype: numpy.ndarray or float
    """
    line_gauss = np.exp(-(v_arr - v_c) ** 2 / 2 / v_g ** 2) / \
                 np.sqrt(2 * np.pi) / v_g * f

    return line_gauss


def model(v_r, k, v_sigma, r, v_g, f, v_c, v_arr):
    """
    compute the full pandisc model

    :param float v_r: float, rotation velocity of the disk
    :param float k: float, gradient of the angular distribution, defining the
        asymmetry of the line, should be in the range (−2/𝜋,2/𝜋)
    :param float v_sigma: float, sigma of the disk velocity dispersion
    :param float r: float, fraction of the integrated flux in the disk component
    :param float v_g: float, sigma of the gaussian peak
    :param float f: float, the integrated flux integrate(f_model dv) of the model
    :param float v_c: float, velocity of the line center
    :param v_arr: array, recording the axis to compute the model
    :type v_arr: numpy.ndarray or float
    :return: array of the computed model, in the same shape as input v_arr
    :rtype: numpy.ndarray or float
    """

    line_disk, line_gauss = model_parts(v_r=v_r, k=k, v_sigma=v_sigma, r=r,
                                        v_g=v_g, f=f, v_c=v_c, v_arr=v_arr)

    return line_disk + line_gauss


def model_parts(v_r, k, v_sigma, r, v_g, f, v_c, v_arr):
    """
    return the disk and gaussian components separately, useful for plotting

    :param float v_r: float, rotation velocity of the disk
    :param float k: float, gradient of the angular distribution, defining the
        asymmetry of the line, should be in the range (−2/𝜋,2/𝜋)
    :param float v_sigma: float, sigma of the disk velocity dispersion
    :param float r: float, fraction of the integrated flux in the disk component
    :param float v_g: float, sigma of the gaussian peak
    :param float f: float, the integrated flux integrate(f_model dv) of the model
    :param float v_c: float, velocity of the line center
    :param v_arr: array, recording the axis to compute the model
    :type v_arr: numpy.ndarray or float
    :return: (disk, gaussian) parts of the model, each part is in the same shape
        as input v_arr
    :rtype: (numpy.ndarray or float, numpy.ndarray or float)
    """

    line_disk = disk(
            v_r=v_r, k=k, v_sigma=v_sigma, f=f * r, v_c=v_c, v_arr=v_arr)
    line_gauss = gaussian(v_g=v_g, f=f * (1 - r), v_c=v_c, v_arr=v_arr)

    return line_disk, line_gauss


def disk_fwhm(v_r, v_sigma):
    """
    estimate the line full width half maximum of the disk component

    :param float v_r: float, rotation velocity of the disk
    :param float v_sigma: float, sigma of the disk velocity dispersion
    :return: float, the estimated line width
    :rtype: float
    """
    return 2 * ((v_r + 0.7 * v_sigma) * (1 - np.exp(-v_r/v_sigma * 1.2)) +
                (v_r + 1.2 * v_sigma) * np.exp(-v_r/v_sigma * 1.8))


def disk_peak_width(v_r, v_sigma):
    """
    estimate the line width between the two peaks of the disk component

    :param float v_r: float, rotation velocity of the disk
    :param float v_sigma: float, sigma of the disk velocity dispersion
    :return: float, the estimated line width between peaks
    :rtype: float
    """

    return (v_r > 1.7 * v_sigma) * (2 * v_r - 1.5 * v_sigma) * \
           (1 - np.exp(-(v_r/v_sigma)**2 + 3))


def disk_peak_flux(v_r, v_sigma, f):
    """
    estimate the flux density of the peaks of the disk component

    :param float v_r: float, rotation velocity of the disk
    :param float v_sigma: float, sigma of the disk velocity dispersion
    :param float f: float, the integrated flux of the disk component
    :return: float, the estimated line width between peaks
    :rtype: float
    """

    return f * (np.arccos(1 - 0.27 * v_sigma/v_r)/np.pi/v_sigma *
                (1 - np.exp(-v_r/v_sigma*1.8)) +
                (1/np.sqrt(2*np.pi)/v_sigma - .013) *
                np.exp(-(v_r/v_sigma)**2/2))


def w50m(v_r, v_sigma, r, v_g):
    """
    estimate the W50 of the line based on the line model

    :param float v_r: float, rotation velocity of the disk
    :param float v_sigma: float, sigma of the disk velocity dispersion
    :param float r: float, fraction of the integrated flux in the disk component
    :param float v_g: float, sigma of the gaussian peak
    :return: float, W50 model
    :rtype: float
    """

    disk_width = disk_fwhm(v_r, v_sigma)
    peak_width = disk_peak_width(v_r, v_sigma)
    disk_flux = disk_peak_flux(v_r, v_sigma, r)
    gauss_flux = (gaussian(v_g, 1, 0, disk_width / 2) +
                  gaussian(v_g, 1, 0, peak_width / 2)) * (1 - r) / 2

    gauss_frac = (gauss_flux/(disk_flux + gauss_flux))
    disk_frac = 1 - gauss_frac
    w50m = disk_frac * disk_width + gauss_frac * 2.355 * v_g

    return w50m
