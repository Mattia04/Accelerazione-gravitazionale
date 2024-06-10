import functools
from typing import Callable, Tuple

import numpy as np
import scipy.stats as st
from scipy import optimize


class Utils:
    @staticmethod
    def iterate_func_through_keys(dataset_keys: Tuple[str], func: Callable) -> None:
        for key in dataset_keys:
            func(key)

    @staticmethod
    def calculate_l_eq(raggio, altezza, lunghezza_filo) -> float:
        """
        Dato che il filo non è di metallo la sua massa è più che trascurabile,
        tuttavia, io ho trascurato anche il momento d'inerzia dell'asta centrale (supporto)
        perché diventava un bordello la formula.
        Inoltre questa formula assume una massa uniforme per tutto.

        args:
            raggio: il raggio del cilindro con le masse
            altezza: l'altezza totale del cilindro con le masse
            lunghezza_filo: la lunghezza del filo
        return:
            l_eq: del pendolo da usare in T = 2pi sqrt(l_eq/g)
        Note:
            non ci sono unità standard quindi vi ritorna la stessa unità che mettete
        """
        L = lunghezza_filo + altezza / 2
        return L + (4 * raggio**2 + altezza**2 / (16 * L))

    staticmethod

    def calculate_l_eq_preciso(
        massa_s, massa_r, raggio_s, raggio_r, altezza_s, altezza_r, lunghezza_filo
    ) -> float:
        """Come calculate_l_eq, solo che viene considerata l'asta del supporto

        args:
            massa_s: massa del supporto (solo l'asta con gancio)
            massa_r: massa di tutto il resto
            raggio_s: raggio del "cilindro" del supporto
            raggio_r: raggio del cilindro delle masse
            altezza_s: altezza del supporto
            altezza_r: altezza del cilindro delle masse
            lunghezza_filo: lunghezza del filo
        """
        m_tot = massa_s + massa_r

        cm = (massa_r * altezza_r + massa_s * altezza_s) / (m_tot)  # ! dal basso

        I_cm = (
            massa_r * (raggio_r**2 / 4 + altezza_r**2 / 16)
            + massa_s * (raggio_s**2 / 4 + altezza_s**2 / 16)
            + m_tot * (altezza_s - cm - altezza_r / 2)
            # ? questo ultimo termine non so se c'è o no,
            # ? sarebbe la correzione del I_cm rispetto al supporto
        )

        L = lunghezza_filo + altezza_s - cm

        return (I_cm + m_tot * L**2) / (m_tot * L)

    # ! this does not work
    def calculate_min_std_dev(periods: np.ndarray) -> Tuple[float]:
        """calculate the minimal standard deviation of the dataset (pag. 37 pdf)"""
        if not isinstance(periods, np.ndarray):
            raise ValueError("The list passed should be a numpy array")

        N = len(periods)
        M = int(2 / 3 * N)  # the best value is at 2/3 of the total
        K = N - M + 1

        T_mean = np.mean(
            [np.mean([period for period in periods[i : M + i + 1]]) for i in range(K)]
        )

        s_k = np.sqrt(
            sum(np.square(np.mean(periods[i : M + i + 1]) - T_mean) for i in range(M))
            / (M - 1)
        )

        s_k_media = s_k / np.sqrt(M)

        return T_mean, s_k, s_k_media


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
        def test_func(x, a, b, c, d, e, phi):
            return a * np.exp(b * (x - phi)) + c * np.exp(d * (x - phi)) + e

        # *

        if sigma_y == -1:
            params, params_cov = optimize.curve_fit(
                test_func,
                x_data,
                y_data,
                p0=[0.1, -0.005, 1.45, -1e-6, 2, 0],
                maxfev=10000,
            )
        else:
            params, params_cov = optimize.curve_fit(
                test_func,
                x_data,
                y_data,
                p0=[1, 1, 0.1, 0.1, 0, 0],
                sigma=sigma_y,
                maxfev=5000,
            )

        return params, np.diag(params_cov)

    @staticmethod
    def chi2(expected, observed, sigma) -> Tuple[float]:
        if not all(isinstance(i, int) for i in [expected, observed, sigma]):
            raise ValueError("All arguments should be numpy arrays")
        return np.sum(np.divide(np.square(expected - observed), np.square(sigma)))

    @staticmethod
    def t_student():
        """sta roba va fatta a mano"""
        pass
