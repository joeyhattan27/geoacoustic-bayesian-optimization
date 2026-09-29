import numpy as np

from .config import (
    DIM,
    FREQUENCIES,
    TRUE_PARAMETERS,
    normalized_to_physical,
)


def normalized_covariance(
    pressure,
):
    pressure = np.asarray(
        pressure,
        dtype=np.complex128,
    )

    norm = np.linalg.norm(pressure)

    if norm == 0:
        raise ValueError("Cannot normalize a zero pressure field")

    q = pressure / norm

    return np.outer(
        q,
        q.conj(),
    )


def bartlett_mismatch(
    covariance,
    replica,
):
    replica = np.asarray(
        replica,
        dtype=np.complex128,
    )

    norm = np.linalg.norm(replica)

    if norm == 0:
        raise ValueError("Cannot evaluate a zero replica")

    w = replica / norm

    match = np.vdot(
        w,
        covariance @ w,
    )

    value = np.trace(covariance) - match

    value = float(np.real(value))

    # Numerical roundoff can produce
    # tiny negative values near zero.
    return max(
        value,
        0.0,
    )


class GeoacousticObjective:
    def __init__(
        self,
        acoustic_model,
    ):
        self.acoustic_model = acoustic_model

        self.observed_covariances = self._generate_observed_covariances()

    def _generate_observed_covariances(
        self,
    ):
        covariances = []

        for frequency in FREQUENCIES:
            pressure = self.acoustic_model.run(
                TRUE_PARAMETERS,
                frequency,
            )

            covariance = normalized_covariance(pressure)

            covariances.append(covariance)

        return covariances

    def evaluate_physical(
        self,
        parameters,
    ):
        parameters = np.asarray(
            parameters,
            dtype=float,
        )

        if parameters.shape != (DIM,):
            raise ValueError(f"Expected {DIM} physical parameters")

        mismatches = []

        for (
            frequency,
            covariance,
        ) in zip(
            FREQUENCIES,
            self.observed_covariances,
        ):
            replica = self.acoustic_model.run(
                parameters,
                frequency,
            )

            mismatch = bartlett_mismatch(
                covariance,
                replica,
            )

            mismatches.append(mismatch)

        return float(np.prod(mismatches))

    def __call__(
        self,
        x_normalized,
    ):
        x = np.asarray(
            x_normalized,
            dtype=float,
        )

        x = np.atleast_2d(x)

        if x.shape[1] != DIM:
            raise ValueError(f"Expected n x {DIM}, got {x.shape}")

        if np.any(x < 0.0) or np.any(x > 1.0):
            raise ValueError("Normalized parameters must stay inside [0, 1]")

        physical = normalized_to_physical(x)

        values = [self.evaluate_physical(parameters) for parameters in physical]

        return np.asarray(
            values,
            dtype=float,
        ).reshape(-1, 1)
