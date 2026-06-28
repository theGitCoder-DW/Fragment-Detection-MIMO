from pathlib import Path
import numpy as np

data_src_IP1 = Path("C:/Research_Project/Fragment-Detection-MIMO/data/MiotyFM_EU1.iq")
data_src_IP2 = Path("C:/Research_Project/Fragment-Detection-MIMO/data/MiotyFM_EU2.iq")

def load_iq(filename: str):

    filename = Path("C:/Research_Project/Fragment-Detection-MIMO/data") / f"{filename}"
    raw = np.fromfile(filename, dtype=np.float32)
    print(raw)
    return raw[0::2] + 1j * raw[1::2]

load_iq("MiotyFM_EU1.iq")