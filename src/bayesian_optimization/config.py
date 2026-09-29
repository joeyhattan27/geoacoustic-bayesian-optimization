from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
CTD_PATH = ROOT_DIR / "data" / "i9605.prn"
RESULTS_DIR = ROOT_DIR / "results"
KRAKEN_TMP_DIR = ROOT_DIR / ".kraken_tmp"

FREQUENCIES = np.array([148.0, 235.0, 388.0])

PARAMETER_NAMES = (
    "source_range_km",
    "source_depth_m",
    "array_tilt_deg",
    "water_depth_m",
    "sediment_thickness_m",
    "sediment_top_speed_mps",
    "sediment_speed_change_mps",
)

LOWER_BOUNDS = np.array([
    0.8,
    60.0,
    -3.0,
    212.0,
    15.0,
    1560.0,
    10.0,
])

UPPER_BOUNDS = np.array([
    1.3,
    80.0,
    3.0,
    222.0,
    25.0,
    1580.0,
    30.0,
])

TRUE_PARAMETERS = np.array([
    1.07,
    71.1,
    2.0,
    217.0,
    23.0,
    1572.3,
    20.7,
])

RECEIVER_DEPTHS = np.array([
    94.125,
    99.755,
    105.38,
    111.0,
    116.62,
    122.25,
    127.88,
    139.12,
    144.74,
    150.38,
    155.99,
    161.62,
    167.26,
    172.88,
    178.49,
    184.12,
    189.76,
    195.38,
    200.99,
    206.62,
    212.25,
])

BASE_WATER_DEPTH = 217.0

SEDIMENT_DENSITY = 1.76
SEDIMENT_ATTENUATION = 0.2

MUDROCK_TOP_SPEED = 1881.0
MUDROCK_BOTTOM_SPEED = 3245.0
MUDROCK_DENSITY = 2.06
MUDROCK_ATTENUATION = 0.06
MUDROCK_THICKNESS = 800.0

BEDROCK_SPEED = 5200.0
BEDROCK_DENSITY = 2.66
BEDROCK_ATTENUATION = 0.02

CLOW = 1400.0
CHIGH = 1800.0

DIM = len(PARAMETER_NAMES)

TOTAL_EVALS = 100
N_INIT = 64
SEEDS = list(range(90, 140))

NUM_RESTARTS = 40
RAW_SAMPLES = 1024

KERNELS = ("rbf", "matern15", "matern25")
ACQUISITION = "logei"
FIT_METHOD = "mll"
ENSEMBLE_WEIGHTS = np.full(len(KERNELS), 1.0 / len(KERNELS))

SUCCESS_THRESHOLDS = (1e-2, 1e-3, 1e-4)


def normalized_to_physical(x):
    x = np.asarray(x, dtype=float)
    return LOWER_BOUNDS + x * (UPPER_BOUNDS - LOWER_BOUNDS)


def physical_to_normalized(x):
    x = np.asarray(x, dtype=float)
    return (x - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)
