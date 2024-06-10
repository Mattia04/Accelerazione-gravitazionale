import numpy as np


def calculate_moving_average(periods: np.ndarray):
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
