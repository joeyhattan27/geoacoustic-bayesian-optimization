import numpy as np

from tritonoa.at.models.kraken.runner import run_kraken

from .config import (
    BASE_WATER_DEPTH,
    BEDROCK_ATTENUATION,
    BEDROCK_DENSITY,
    BEDROCK_SPEED,
    CHIGH,
    CLOW,
    CTD_PATH,
    KRAKEN_TMP_DIR,
    MUDROCK_ATTENUATION,
    MUDROCK_BOTTOM_SPEED,
    MUDROCK_DENSITY,
    MUDROCK_THICKNESS,
    MUDROCK_TOP_SPEED,
    RECEIVER_DEPTHS,
    SEDIMENT_ATTENUATION,
    SEDIMENT_DENSITY,
)


def load_sound_speed_profile(path=CTD_PATH):
    if not path.exists():
        raise FileNotFoundError(f"CTD file not found: {path}")

    data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Unexpected CTD format in {path}")

    depth = data[:, 0]
    sound_speed = data[:, 3]

    if len(depth) < 2:
        raise ValueError("CTD profile is too short")

    if np.any(np.diff(depth) < 0):
        raise ValueError("CTD depths must be sorted")

    return depth, sound_speed


class AcousticModel:
    def __init__(self, ctd_path=CTD_PATH):
        self.ctd_depth, self.ctd_speed = load_sound_speed_profile(ctd_path)

    def _water_profile(self, water_depth):
        keep = self.ctd_depth < water_depth

        depth = self.ctd_depth[keep]
        speed = self.ctd_speed[keep]

        bottom_speed = np.interp(
            water_depth,
            self.ctd_depth,
            self.ctd_speed,
            left=self.ctd_speed[0],
            right=self.ctd_speed[-1],
        )

        depth = np.append(
            depth,
            water_depth,
        )

        speed = np.append(
            speed,
            bottom_speed,
        )

        return (
            depth.tolist(),
            speed.tolist(),
        )

    def build_environment(
        self,
        parameters,
        frequency,
    ):
        (
            source_range,
            source_depth,
            tilt,
            water_depth,
            sediment_thickness,
            sediment_top_speed,
            sediment_speed_change,
        ) = np.asarray(
            parameters,
            dtype=float,
        )

        water_z, water_c = self._water_profile(water_depth)

        depth_shift = water_depth - BASE_WATER_DEPTH

        receiver_depths = RECEIVER_DEPTHS + depth_shift

        sediment_bottom = water_depth + sediment_thickness

        sediment_bottom_speed = sediment_top_speed + sediment_speed_change

        mudrock_bottom = sediment_bottom + MUDROCK_THICKNESS

        environment = {
            "title": f"swellex_{frequency:.0f}",
            "model": "KRAKENC",
            "freq": float(frequency),
            "tmpdir": KRAKEN_TMP_DIR,
            "layerdata": [
                {
                    "z": water_z,
                    "c_p": water_c,
                    "rho": 1.0,
                },
                {
                    "z": [
                        water_depth,
                        sediment_bottom,
                    ],
                    "c_p": [
                        sediment_top_speed,
                        sediment_bottom_speed,
                    ],
                    "rho": SEDIMENT_DENSITY,
                    "a_p": SEDIMENT_ATTENUATION,
                },
                {
                    "z": [
                        sediment_bottom,
                        mudrock_bottom,
                    ],
                    "c_p": [
                        MUDROCK_TOP_SPEED,
                        MUDROCK_BOTTOM_SPEED,
                    ],
                    "rho": MUDROCK_DENSITY,
                    "a_p": MUDROCK_ATTENUATION,
                },
            ],
            "bot_opt": "A",
            "bot_c_p": BEDROCK_SPEED,
            "bot_rho": BEDROCK_DENSITY,
            "bot_a_p": BEDROCK_ATTENUATION,
            "clow": CLOW,
            "chigh": CHIGH,
            "src_z": source_depth,
            "rec_z": receiver_depths.tolist(),
            "rec_r": source_range,
            "tilt": tilt,
            "z_pivot": water_depth,
        }

        return environment

    def run(
        self,
        parameters,
        frequency,
    ):
        environment = self.build_environment(
            parameters,
            frequency,
        )

        pressure = run_kraken(environment)

        pressure = np.asarray(
            pressure,
            dtype=np.complex128,
        )

        pressure = np.squeeze(pressure)

        if pressure.ndim != 1:
            raise RuntimeError(f"Expected a 1D pressure field, got {pressure.shape}")

        if pressure.size != RECEIVER_DEPTHS.size:
            raise RuntimeError(
                f"Expected {RECEIVER_DEPTHS.size} receiver values, got {pressure.size}"
            )

        if not np.all(np.isfinite(pressure)):
            raise RuntimeError("KRAKENC returned non-finite values")

        if np.linalg.norm(pressure) == 0:
            raise RuntimeError("KRAKENC returned a zero pressure field")

        return pressure
