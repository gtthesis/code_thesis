import numpy as np
from scipy.optimize import curve_fit

p0 = [2.7042514764602656, 81.32295675879257, 21.844197482382192, 55.543523790816614,
      85.70171449210308, 54.06407157464707,
      181.2645006824145, 56.02454782133164,
      0.05]     # Parameter guesses k=2.7, l=81...

bounds = ([2.6, 80, -10, 30,
           80, 0,
           160, 30,
           0],

          [6, 120, 30, 100,
           130, 100,
           210, 100,
           0.2])    # Parameter bounds k=2.6, l=80...


def weibull(x, k, l, mu):
    filter = np.zeros_like(x)
    idx = int(np.clip(mu, 0, len(x) - 1))
    filter[idx:] = 1        # This is to zeroize the weibull where it would be undefined by x-mu < 0. The filter multiplies weibull by 1 where it is valid (x-mu>0), else multiplied by 0.
    return np.nan_to_num(filter * ((k / l) * (((x - mu) / l) ** (k - 1)) * np.exp(-((x - mu) / l) ** k)))


def triwei(x, k, l, mu1, P1, mu2, P2, mu3, P3, B):
    return (P1 * weibull(x, k, l, mu1)) + (P2 * weibull(x, k, l, mu2)) + (P3 * weibull(x, k, l, mu3)) + B  # Mixture of weibulls, our model.


def fit_triwei(x):
    xi = np.linspace(0, len(x) - 1, len(x))     # time x coordinates to units
    popt, pcov = curve_fit(triwei, xi, x.values.squeeze(), p0=p0, bounds=bounds, max_nfev=10000)    # popt best parameters found
    return triwei(xi, *popt)


def get_triwei_params(x):
    xi = np.linspace(0, len(x) - 1, len(x))     # time x coordinates to units
    popt, pcov = curve_fit(triwei, xi, x.values.squeeze(), p0=p0, bounds=bounds, max_nfev=10000)
    if any(p == b for p, b in zip(popt, bounds[0])) or any(p == b for p, b in zip(popt, bounds[1])):
        assert("Parameters reached bounds")
    return popt
