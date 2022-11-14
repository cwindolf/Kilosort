%%
% full Kilosort2.5 runner fn
% -- with --
% replaced interpolation logic so that you can use your
% own displacement estimate
% the displacement estimate will be loaded from the path pNpy
% the entries in the displacement estimate are assumed to be sampled
% at the rate pfs (sampling frequency in 1/s)
% the displacement estimate is assumed to start at time tStart
%%
function [] = ks25(dataDir, scratchDir, configFile, chanMapFile, tStart, tEnd, NchanTOT, pNpy, pfs)

path0 = fileparts(mfilename('fullpath'));
addpath(genpath(path0)) % path to kilosort folder

ops.trange    = [tStart tEnd]; % time range to sort
ops.NchanTOT  = NchanTOT; % total number of channels in your recording

run(configFile)
ops.fproc   = fullfile(scratchDir, 'temp_wh.dat'); % proc file on a fast SSD
ops.chanMap = chanMapFile;

%% this block runs all the steps of the algorithm
fprintf('Looking for data inside %s \n', dataDir)

% main parameter changes from Kilosort2 to v2.5
ops.sig        = 20;  % spatial smoothness constant for registration
ops.fshigh     = 300; % high-pass more aggresively
ops.nblocks    = 1; % blocks for registration. 0 turns it off, 1 does rigid registration. Replaces "datashift" option. 

% is there a channel map file in this folder?
fs = dir(fullfile(dataDir, 'chan*.mat'));
if ~isempty(fs)
    ops.chanMap = fullfile(dataDir, fs(1).name);
end

% find the binary file
fs          = [dir(fullfile(dataDir, '*.bin')) dir(fullfile(dataDir, '*.dat'))];
ops.fbinary = fullfile(dataDir, fs(1).name);

% load up displacement estimate using npy-matlab
displacement = readNPY(pNpy);

% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);

%
% *REPLACED* NEW STEP TO DO DATA REGISTRATION
% 0 means no correction here
% we still need to do this because KS needs certain things this sets
% but, it ignores dshift after the fact so it's ok to run it anyway!
rez = datashift2(rez, 0); % last input is for shifting data
% sneak in and apply our displacement estimate
interpolate_binary_with_drift(rez, displacement, pfs);

% ORDER OF BATCHES IS NOW RANDOM, controlled by random number generator
iseed = 1;
                 
% main tracking and template matching algorithm
rez = learnAndSolve8b(rez, iseed);

% OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
% See issue 29: https://github.com/MouseLand/Kilosort/issues/29
%rez = remove_ks2_duplicate_spikes(rez);

% final merges
rez = find_merges(rez, 1);

% final splits by SVD
rez = splitAllClusters(rez, 1);

% decide on cutoff
rez = set_cutoff(rez);
% eliminate widely spread waveforms (likely noise)
rez.good = get_good_units(rez);

fprintf('found %d good units \n', sum(rez.good>0))

% write to Phy
% fprintf('Saving results to Phy  \n')
rezToPhy(rez, dataDir);

%% if you want to save the results to a Matlab file...

% discard features in final rez file (too slow to save)
rez.cProj = [];
rez.cProjPC = [];

% final time sorting of spikes, for apps that use st3 directly
[~, isort]   = sortrows(rez.st3);
rez.st3      = rez.st3(isort, :);

% Ensure all GPU arrays are transferred to CPU side before saving to .mat
rez_fields = fieldnames(rez);
for i = 1:numel(rez_fields)
    field_name = rez_fields{i};
    if(isa(rez.(field_name), 'gpuArray'))
        rez.(field_name) = gather(rez.(field_name));
    end
end

% save final results as rez2
fprintf('Saving final results in rez2  \n')
fname = fullfile(dataDir, 'rez2.mat');
save(fname, 'rez', '-v7.3');

end
