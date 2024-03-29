import functools
from typing import Callable, Tuple

import numpy as np
import scipy.stats as st
from scipy import optimize


class Utils:
    def iterate_func_through_keys(dataset_keys: Tuple[str], func: Callable) -> None:
        for key in dataset_keys:
            func(key)

    def calculate_cm():
        pass

    def calculate_min_std_dev():
        """calculate the minimal standard deviation of the dataset (pag. 37 pdf)"""
        pass


class Statistics:
    @staticmethod
    def calculate_conf_bound(
        mean: float, sigma: float, confidence: float = 0.95
    ) -> Tuple[float]:
        # ! Note the sigma should be the sigma of the mean if it's a mean
        return st.norm.interval(confidence, loc=mean, scale=sigma)

    @staticmethod
    def linfit(x, y, weights=-1) -> Tuple[float]:
        if weights == -1:
            weights = np.ones(len(x))

        coeff, cov = np.polyfit(x, y, 1, w=weights, cov=True)
        return *coeff, *np.diag(cov)

    @staticmethod
    def expfit(x, y, weights=-1):
        if weights == -1:
            A, B, eA, eB = Statistics.linfit(x, np.log(y))
        else:
            A, B, eA, eB = Statistics.linfit(x, np.log(y), weights=np.log(weights))
        return A, np.exp(B), eA, np.exp(eB)

    @staticmethod
    def logfit(x, y, weights=-1):
        if weights == -1:
            A, B, eA, eB = Statistics.linfit(np.log(x), y)
        else:
            A, B, eA, eB = Statistics.linfit(np.log(x), y, weights=weights)
        return np.exp(A), B, np.exp(eA), eB

    @staticmethod
    def sinfit(
        x_data: np.array, y_data: np.array, sigma_y: np.array = -1
    ) -> Tuple[float]:
        # *
        def test_func(x, a, b, c, d):
            return a * np.sin(b * x + c) + d

        # *

        if sigma_y == -1:
            params, params_cov = optimize.curve_fit(
                test_func, x_data, y_data, p0=[1, 1, 0, 0]
            )
        else:
            params, params_cov = optimize.curve_fit(
                test_func, x_data, y_data, p0=[1, 1, 0, 0], sigma=sigma_y
            )

        return *params, *np.diag(params_cov)

    @staticmethod
    def double_expfit(
        x_data: np.array, y_data: np.array, sigma_y: np.array = -1
    ) -> Tuple[float]:
        # *
        def test_func(x, a, b, c, d, e):
            return a * np.exp(b * x) + c * np.exp(d * x) + e

        # *

        if sigma_y == -1:
            params, params_cov = optimize.curve_fit(
                test_func, x_data, y_data, p0=[1, -1, 1, -0.0001, 0], maxfev=5000
            )
        else:
            params, params_cov = optimize.curve_fit(
                test_func,
                x_data,
                y_data,
                p0=[1, 1, 0.1, 0.1, 0],
                sigma=sigma_y,
                maxfev=5000,
            )

        return *params, *np.diag(params_cov)

    @staticmethod
    def chi2(expected, observed, sigma) -> Tuple[float]:
        if not all(isinstance(i, int) for i in [expected, observed, sigma]):
            raise ValueError("All arguments should be numpy arrays")
        return np.sum(np.divide(np.square(expected - observed), np.square(sigma)))

    @staticmethod
    def t_student():
        pass
