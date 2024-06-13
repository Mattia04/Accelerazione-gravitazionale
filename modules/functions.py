import numpy as np
from scipy.optimize import curve_fit


def calc_moving_average(periods: np.ndarray):
    if len(np.shape(periods)) == 1:
        return calc_list_MA(periods)
    window_size = (periods.shape[1] * 2) // 3

    def calculate_window(window: np.ndarray):
        mean = np.mean(window, axis=1)
        window_std = np.std(window, axis=1, ddof=1)
        uncertainty = np.divide(window_std, np.sqrt(window_size))

        return mean, uncertainty

    shape_result = (periods.shape[1] - window_size +1, periods.shape[0])
    averages, uncertainties = np.zeros(shape=shape_result), np.zeros(shape=shape_result)
    for i in range(periods.shape[1] - window_size + 1):
        current_window = periods[:, i : i + window_size]
        avg, err = calculate_window(current_window)
        averages[i] += avg
        uncertainties[i] += err

    average = np.mean(averages, axis=0)
    uncertainty = np.std(averages, axis=0, ddof=1)/ np.sqrt(window_size)

    return average, uncertainty

def calc_list_MA(periods: np.ndarray):
    n_periods = len(periods)
    window_size = (n_periods * 2) // 3

    def calculate_window(window: np.ndarray):
        mean = np.mean(window)
        window_std = np.std(window, ddof=1)
        uncertainty = np.divide(window_std, np.sqrt(window_size))

        return mean, uncertainty

    shape_result = (n_periods - window_size +1)
    averages, uncertainties = np.zeros(shape=shape_result), np.zeros(shape=shape_result)
    for i in range(n_periods- window_size + 1):
        current_window = periods[i : i + window_size]
        avg, err = calculate_window(current_window)
        averages[i] += avg
        uncertainties[i] += err

    average = np.mean(averages)
    uncertainty = np.std(averages, ddof=1)/ np.sqrt(window_size)

    return average, uncertainty


def calc_reg_lin(X, Y, err):
    retta = lambda x, a, b: a * x + b

    if len(np.shape(X)) == 1:
        params, cov = curve_fit(retta, X, Y, sigma=err, absolute_sigma=True)
        std = np.sqrt(np.diag(cov))

        return params, std
    # ! not sure about that
    elif len(np.shape(X)) == 2:
        params, std = [], []
        for i in range(X.shape()[0]):
            param, cov = curve_fit(retta, X[:, i], Y[:, i], sigma=err[:, i], absolute_sigma=True)
            params.append(param)
            std.append(np.sqrt(np.diag(cov)))

        return params, std
    else:
        raise ValueError("Not yet implemented for arrays of dimensions greater than 2")

def calc_cm():
    # Calculate center of mass
    massa_gancio = 19.88  # g
    massa_masse = 79.56  # g
    massa_tot = 19.88 + 79.56  # g

    altezza_base_gancio = 71.38  # mm
    altezza_masse = 21.28  # mm

    return (
        altezza_base_gancio * massa_gancio + altezza_masse * massa_masse
    ) / massa_tot
