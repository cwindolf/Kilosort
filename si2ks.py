"""Preprocessed spikeinterface binary -> KS style binary -> KS"""
import numpy as np
import spikeinterface.core as sc
import spikeinterface.preprocessing as spre
from scipy.io import savemat
from pathlib import Path
import subprocess
import shutil
import tempfile

def si2ks(si_folder, ks_folder):
    rec = sc.read_binary_folder(si_folder)
    rec = spre.scale(rec, gain=200.0, dtype=np.int16)
    shutil.rmtree(ks_folder)
    rec = rec.save_to_folder(ks_folder)

    # also dump the chan map
    chanmap = dict(
        chanMap=np.arange(1, rec.get_num_channels() + 1)[:, None],
        chanMap0ind=np.arange(rec.get_num_channels())[:, None],
        connected=np.ones(rec.get_num_channels())[:, None],
        xcoords=rec.get_channel_locations()[:, 0, None],
        ycoords=rec.get_channel_locations()[:, 1, None],
        kcoords=np.ones(rec.get_num_channels())[:, None],
    )
    cmpath = Path(ks_folder) / f"chanMap.mat"
    savemat(cmpath, chanmap)
    
    return rec, cmpath

def run_ks(rec, cmpath, ks_folder, cache_directory, full_ks=False):
    this_dir = Path(__file__).parent
    config_m = this_dir / "configFiles/configFile384.m"
    if full_ks:
        cmd = f"ks25('{ks_folder}', '{cache_directory}', '{config_m}', '{cmpath}', 0, Inf, {rec.get_num_channels()})"
    else:
        cmd = f"main_kilosort('{ks_folder}', '{cache_directory}', '{config_m}', '{cmpath}', 0, Inf, {rec.get_num_channels()}, 100, 5)"
    
    subprocess.run(
        f'ml load matlab/2022b && cd {this_dir} && matlab -nodisplay -nosplash -r "{cmd}; exit;"',
        shell=True,
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--si-folder")
    ap.add_argument("--ks-folder")
    ap.add_argument("--cache-dir-parent", default="/local")
    ap.add_argument("--full-ks", action="store_true")

    args = ap.parse_args()

    rec, cmpath = si2ks(args.si_folder, args.ks_folder)
    with tempfile.TemporaryDirectory(dir=args.cache_dir_parent) as tempdir:
        run_ks(rec, cmpath, args.ks_folder, cache_directory=tempdir, full_ks=args.full_ks)