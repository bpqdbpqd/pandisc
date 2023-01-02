"""
Functions for MCMC fit formulation on H I spectra. If you are interested in
applying the model on other spectral line, the accepted range of 25 km/s might
be too small, so you may want to change pandisc.SIGMA_MAX to higher value
"""

from .model import *


V_SIGMA_MAX = 25  # default maximal acceptable value of v_sigma in km/s


def model_mcmc(para_mcmc, v_arr):
    """
    helper function for MCMC fit, to generate line model from the parameters
    used in fitting

    :param para_mcmc: list or tuple or array, containing the seven parameters
        used for MCMC fit, in the oder of (lg_v_r, k, v_sigma, r, lg_v_g, f, v_c)
    :type para_mcmc: list or tuple or numpy.ndarray
    :param v_arr: array, recording the axis to compute the model
    :type v_arr: numpy.ndarray or float
    :return: array of the computed model, in the same shape as input v_arr
    :rtype: numpy.ndarray or float
    """

    lg_v_r, k, v_sigma, r, lg_v_g, f, v_c = para_mcmc
    v_r, v_g = 10**lg_v_r, 10**lg_v_g

    return model(v_r=v_r, k=k, v_sigma=v_sigma, r=r, v_g=v_g, f=f,
                 v_c=v_c, v_arr=v_arr)


def model_mcmc_parts(para_mcmc, v_arr):
    """
    return the disk and gaussian components separately using the set of the
    parameters used in MCMC fit

    :param para_mcmc: list or tuple or array, containing the seven parameters
        used for MCMC fit, in the oder of (lg_v_r, k, v_sigma, r, lg_v_g, f, v_c)
    :type para_mcmc: list or tuple or numpy.ndarray
    :param v_arr: array, recording the axis to compute the model
    :type v_arr: numpy.ndarray or float
    :return: (disk, gaussian) parts of the model, each part is in the same shape
        as input v_arr
    :rtype: tuple[numpy.ndarray or float, numpy.ndarray or float]
    """

    lg_v_r, k, v_sigma, r, lg_v_g, f, v_c = para_mcmc
    v_r, v_g = 10**lg_v_r, 10**lg_v_g

    return model_parts(v_r=v_r, k=k, v_sigma=v_sigma, r=r, v_g=v_g, f=f,
                       v_c=v_c, v_arr=v_arr)


def ln_priori_flat(para_mcmc, v_arr):
    """
    a flat priori function defined in the paper, only checking the range of the
    input parameters

    :param para_mcmc: list or tuple or array, containing the seven parameters
        used for MCMC fit, in the oder of (lg_v_r, k, v_sigma, r, lg_v_g, f, v_c)
    :type para_mcmc: list or tuple or numpy.ndarray
    :param numpy.ndarray v_arr: array, recording the axis to compute the model
    :return: float, log e of the priori
    :rtype: float
    """

    lg_v_r, k, v_sigma, r, lg_v_g, f, v_c = para_mcmc
    v_r, v_g = 10**lg_v_r, 10**lg_v_g

    if (5 < v_r < 500) and -2/np.pi < k < 2/np.pi and \
            3 < v_sigma < V_SIGMA_MAX and \
            (0 <= r <= 1) and (8.5 < v_g < 200) and \
            (v_arr.min() < v_c - v_r * r) and \
            (v_c + v_r * r < v_arr.max()) and \
            (v_c - v_r * r > v_arr.min()):
        priori = np.log(1/2 * np.pi/4 / (V_SIGMA_MAX - 3) / 1.4)
    else:
        priori = -np.inf

    return priori


def ln_priori(para_mcmc, v_arr):
    """
    the priori function defined in the paper

    :param para_mcmc: list or tuple or array, containing the seven parameters
        used for MCMC fit, in the oder of (lg_v_r, k, v_sigma, r, lg_v_g, f, v_c)
    :type para_mcmc: list or tuple or numpy.ndarray
    :param numpy.ndarray v_arr: array, recording the axis to compute the model
    :return: float, log e of the priori
    :rtype: float
    """

    lg_v_r, k, v_sigma, r, lg_v_g, f, v_c = para_mcmc
    v_r, v_g = 10**lg_v_r, 10**lg_v_g

    priori = ln_priori_flat(para_mcmc, v_arr)
    if np.isfinite(priori):
        priori += np.log(4/np.pi / 0.568 / 0.786) + \
                  np.log(0.44) * (1 - r)**2 - 3 * abs(k)
        if (r > 0) and \
                (2 * (1 - r)/r/np.sqrt(2 * np.pi) > 10**(lg_v_g - lg_v_r)) and \
                (1/2 + .3 * v_sigma/v_r > 10**(lg_v_g-lg_v_r)):
            priori += np.log(min(1 - (1 - r)/r * np.sqrt(2/np.pi) * (v_r - v_g),
                                 1 - (v_r + .6 * v_sigma)/2/v_g))
    else:
        priori = -np.inf
    return priori


def ln_like(para_mcmc, v_arr, spec, sigma=2.23):
    """
    compute the log likelihood for the model parameter given the input spectrum,
    using a per-channel rms

    :param para_mcmc: list or tuple or array, containing the seven parameters
        used for MCMC fit, in the oder of (lg_v_r, k, v_sigma, r, lg_v_g, f, v_c)
    :type para_mcmc: list or tuple or numpy.ndarray
    :param numpy.ndarray v_arr: array, recording the axis to compute the model
    :param numpy.ndarray spec: array, input spectrum to evaluate likelihood for
        the parameters, must have the same shape as v_arr
    :param float sigma: float, sigma per channel used for evaluating likelihood
    :return: float, log likelihood of the input parameters
    :rtype: float
    """

    line_model = model_mcmc(para_mcmc, v_arr)
    likelihood = - ((spec - line_model)**2 / 2 / sigma**2).sum()

    return likelihood


def ln_prob(para_mcmc, v_arr, spec, sigma=2.23):
    """
    compute the posterior likelihood of the input parameter, by combining
    priori and likelihood

    :param para_mcmc: list or tuple or array, containing the seven parameters
        used for MCMC fit, in the oder of (lg_v_r, k, v_sigma, r, lg_v_g, f, v_c)
    :type para_mcmc: list or tuple or numpy.ndarray
    :param numpy.ndarray v_arr: array, recording the axis to compute the model
    :param numpy.ndarray spec: array, input spectrum to evaluate likelihood for
        the parameters, must have the same shape as v_arr
    :param float sigma: float, sigma per channel used for evaluating likelihood
    :return: float, log posterior likelihood of the input parameters
    :rtype: float
    """

    priori = ln_priori(para_mcmc, v_arr)
    if not np.isfinite(priori):
        return -np.inf
    else:
        return priori + ln_like(para_mcmc, v_arr, spec, sigma)
