from pathlib import Path

import yaml

data =   {
    "RESULTS_DIR": "results",

    # Data files
    "DATA_DIR": "./baselines/alphaEdit/rome/data",
    "STATS_DIR": "./baselines/alphaEdit/rome/data/stats",
    "KV_DIR": "./baselines/alphaEdit/kvs",
    "HPARAMS_DIR": "hparams",
    "REMOTE_ROOT_URL": "https://memit.baulab.info"
}

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
