"""
Functions to compute the pandisc model. For the formulation please refer to
the documentation.
"""

import numpy as np


def pandisc_disk(v_r, k, v_sigma, f, v_c, v_arr):
    """
    compute the co-rotating disk part of the model, the integrated model flux is
    normalized to f

    :param float v_r: float, rotation velocity of the disk
    :param float k: float, gradient of the angular distribution, defining the
        asymmetry of the line, should be in the range (−2/𝜋,2/𝜋)
    :param float v_sigma: float, sigma of the disk velocity dispersion
    :param float f: float, the integrated flux integrate(f_model dv) of the model
    :param float v_c: float, velocity of the line center
    :param numpy.ndarray v_arr: array, recording the axis to compute the model
    :return: array of the computed model
    :rtype: numpy.ndarray
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


def pandisc_gaussian(v_g, f, v_c, v_arr):
    """
    compute the gaussian component of the model, the integrated flux is
    normalized to f

    :param float v_g: float, sigma of the gaussian peak
    :param float f: float, the integrated flux integrate(f_model dv) of the model
    :param float v_c: float, velocity of the line center
    :param numpy.ndarray v_arr: array, recording the axis to compute the model
    :return: array of the computed model
    :rtype: numpy.ndarray
    """
    line_gauss = np.exp(-(v_arr - v_c) ** 2 / 2 / v_g ** 2) / \
                 np.sqrt(2 * np.pi) / v_g * f

    return line_gauss


def pandisc_model(para, v_arr):
    """
    compute the full pandisc model

    :param para: list or tuple or array, containing all seven parameters in the
        order of (v_r, k, v_sigma, r, v_g, f, v_c)
    :type para: list or tuple or numpy.ndarray
    :param numpy.ndarray v_arr: array, recording the axis to compute the model
    :return: array of the computed model
    :rtype: numpy.ndarray
    """

    line_disk, line_gauss = pandisc_model_parts(para, v_arr)

    return line_disk + line_gauss


def pandisc_model_parts(para, v_arr):
    """
    return the disk and gaussian components separately, useful for plotting

    :param para: list or tuple or array, containing all seven parameters in the
        order of (v_r, k, v_sigma, r, v_g, f, v_c)
    :type para: list or tuple or numpy.ndarray
    :param numpy.ndarray v_arr: array, recording the axis to compute the model
    :return: array of the computed model
    :rtype: numpy.ndarray
    """
    v_r, k, v_sigma, r, v_g, f, v_c = para

    line_disk = pandisc_disk(
            v_r=v_r, k=k, v_sigma=v_sigma, f=f * r, v_c=v_c, v_arr=v_arr)
    line_gauss = pandisc_gaussian(v_g=v_g, f=f * (1 - r), v_c=v_c, v_arr=v_arr)

    return line_disk, line_gauss
