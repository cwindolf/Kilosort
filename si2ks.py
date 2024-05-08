"""Preprocessed spikeinterface binary -> KS style binary -> KS"""
import numpy as np
import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from scipy.io import savemat
from pathlib import Path
import subprocess
import shutil
import tempfile

def si2ks(si_folder, ks_folder, geom=None, n_jobs=2, scaleproc=True, spikeglx=False):
    if spikeglx:
        rec = se.read_spikeglx(si_folder)
        geom = rec.get_channel_locations()
        assert si_folder == ks_folder
    elif scaleproc:
        try:
            rec = sc.read_binary_folder(si_folder)
            geom = rec.get_channel_locations()
        except:
            assert geom is not None
            rec = sc.read_binary(next(Path(si_folder).glob("*.bin")), 30000.0, num_channels=len(geom), dtype="float32")
            rec.set_dummy_probe_from_locations(geom)
        
        rec = spre.scale(rec, gain=200.0, dtype=np.int16)
        ks_folder = Path(ks_folder)
        if ks_folder.exists():
            shutil.rmtree(ks_folder)
        rec = rec.save_to_folder(ks_folder, n_jobs=n_jobs, chunk_memory="512M")
    else:
        assert ks_folder == si_folder
        try:
            rec = sc.read_binary_folder(si_folder)
            geom = rec.get_channel_locations()
        except:
            assert geom is not None

    # also dump the chan map
    chanmap = dict(
        chanMap=np.arange(1, len(geom) + 1)[:, None],
        chanMap0ind=np.arange(len(geom))[:, None],
        connected=np.ones(len(geom))[:, None],
        xcoords=geom[:, 0, None],
        ycoords=geom[:, 1, None],
        kcoords=np.ones(len(geom))[:, None],
    )
    cmpath = Path(ks_folder) / f"chanMap.mat"
    savemat(cmpath, chanmap)
    
    return len(geom), cmpath

def run_ks(nc, cmpath, ks_folder, cache_directory, tstart=0, tend="Inf", full_ks=False, nBinsReg=15, depthBin=5, nblocks=1, prefix_cmd='ml load matlab/2022b'):
    this_dir = Path(__file__).parent
    config_m = this_dir / "configFiles/configFile384.m"
    if full_ks:
        print("Full KS with default params!")
        cmd = f"ks25('{ks_folder}', '{cache_directory}', '{config_m}', '{cmpath}', {tstart}, {tend}, {nc})"
    else:
        print("Not running full KS, just registration part")
        cmd = f"main_kilosort('{ks_folder}', '{cache_directory}', '{config_m}', '{cmpath}', {tstart}, {tend}, {nc}, {nBinsReg}, {depthBin}, {nblocks})"
        
    matlab_cmd = f'matlab -nodisplay -nosplash -r "{cmd}; exit;"'
    fullcmd = f"cd {this_dir} && {matlab_cmd}"
    if prefix_cmd:
        fullcmd = f"{prefix_cmd} && {fullcmd}"
    
    subprocess.run(
        fullcmd,
        shell=True,
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--si-folder")
    ap.add_argument("--ks-folder")
    ap.add_argument("--cache-dir-parent", default="/local")
    ap.add_argument("--full-ks", action="store_true")
    ap.add_argument("--nblocks", type=int, default=5, help="keep in mind that this is not the real number...")
    ap.add_argument("--nBinsReg", type=int, default=15)
    ap.add_argument("--depthBin", type=int, default=5)
    ap.add_argument("--geom-npy", type=str, default=None)
    ap.add_argument("--prefix-cmd", type=str, default='ml load matlab/2022b')
    ap.add_argument("--no-scaleproc", action="store_true")
    ap.add_argument("--spikeglx", action="store_true")
    ap.add_argument("--tstart", type=float, default=0)
    ap.add_argument("--tend", type=float, default=np.inf)

    args = ap.parse_args()
    
    print("si2ks.py")

    tstart = str(args.tstart)
    tend = str(args.tend)
    if args.tend == np.inf:
        tend = "Inf"
    
    geom = None
    if args.geom_npy:
        geom = np.load(args.geom_npy)

    nc, cmpath = si2ks(args.si_folder, args.ks_folder, geom=geom, scaleproc=not args.no_scaleproc, spikeglx=args.spikeglx)
    with tempfile.TemporaryDirectory(dir=args.cache_dir_parent) as tempdir:
        run_ks(nc, cmpath, args.ks_folder, tstart=tstart, tend=tend, cache_directory=tempdir, full_ks=args.full_ks, nblocks=args.nblocks, nBinsReg=args.nBinsReg, depthBin=args.depthBin, prefix_cmd=args.prefix_cmd)
