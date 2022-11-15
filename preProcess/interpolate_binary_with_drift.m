% routines for modifying KS's internal temp_wh.dat binary file
% according to displacement estimates whose time domain does not
% match up with kilosort's internal batching
% these modify temp_wh.dat in place!
% adapted from datashift2 and shift_batch_on_disk2

function interpolate_binary_with_drift(rez, p, pfs, ysamp)
% this loop will use a drift estimate to interpolate
% the whitened binary that KS uses on disk
% I *think* the way to use this is to run it just before
% datashift2, and then make sure datashift2 is called with
% the second argument equal to 0.
% but I have not tested this much yet!

% ysamp is the depth domain of p, if p is a matrix (nonrigid case)
% if it's rigid, then this is nonsense and you don't need to supply this argument
if nargin < 4
ysamp = 0;
end

% notes on time domain of p
% we assume (!) that p's time domain starts at the sample ops.tstart in the original
% binary file, which is defined in preprocessDataSub, or equivilantly at the start
% of kilosorts temp_wh.dat.
% then, each entry in p comes 1/pfs seconds later.
% we will discretize this time domain to samples in the for loop below.
samples_per_bin = rez.ops.fs / pfs;

% dprev: what is dprev?
% in datashift2, each batch is loaded with a `ntbuff` padding to the right
% then, when shifting batch i, we do a smoothing average of batch i's
% first `ntbuff` samples with these padding samples from the last batch.
% we start using 0s, which is a little weird since it'll suppress early
% spikes, but that's what they do.
ntbuff = rez.ops.ntbuff;  % this is 64 I think pretty much always
dprev = gpuArray.zeros(ntbuff, rez.ops.Nchan, 'single');

if samples_per_bin < ntbuff
    disp("samples_per_bin < ntbuff, which could be a big problem...");
end


% loop: for each time bin of p, apply the kriging directly to the raw binary file
% start of batch in samples
fprintf("Interpolating data (20 dots total): ")
batchstart = 0;
% bugs when using this, actually a segfault that was confusing to debug... not sure why
% max_t_samples = rez.ops.sampsToRead;
max_t_samples = rez.ops.Nbatch * rez.ops.NT;
tic
for i = 1:size(p,1)
    % a sort of progress bar with 20 ticks
    if ~mod(i, floor(size(p, 1) / 20))
        fprintf('.')
    end

    % end of current batch in samples
    batchend = min(max_t_samples, round(i * samples_per_bin));

    % how many frames to read
    % if we read N frames from 0, then the first frame of the next
    % batch is N. in other words this is a 0-indexed situation.
    batchlen = 1 + batchend - batchstart;

    dprev = shift_batch_on_disk_modified(rez, batchstart, batchlen, p(i), ysamp, rez.ops.sig, dprev, ntbuff);

    % for contiguous batches
    batchstart = batchend;

    if batchend == max_t_samples
        break
    end
end
% close off the progress bar
fprintf('\n')
toc

if batchend < max_t_samples
    disp("p's time domain was shorter than [tStart tEnd] which will probably be bad");
end

end

function [dprev, dat_cpu, dat, shifts] = ...
    shift_batch_on_disk_modified(rez, batchstart, batchlen, shifts, ysamp, sig, dprev, ntb)
% this is a reimplementation of shift_batch_on_disk2
% register one batch of a whitened binary file

offset = 2 * rez.ops.Nchan * batchstart; % binary file offset in bytes

% upsample the shift for each channel using interpolation
if length(ysamp)>1
    shifts = interp1(ysamp, shifts, rez.yc, 'makima', 'extrap');
end

% load the batch
fid = fopen(rez.ops.fproc, 'r+');
fseek(fid, offset, 'bof');
dat = fread(fid, [rez.ops.Nchan batchlen+ntb], '*int16')';

% since we are loading batches differently, we are on our own
% when it comes to the padding logic. the other method can
% assume all chunks are exactly the same time since KS makes
% sure of this.
nsampcurr = size(dat,2);
if nsampcurr < batchlen+ntb
    % pad just so the code runs, but the padding will be thrown out below
    dat(:, nsampcurr+1:batchlen+ntb) = repmat(dat(:,nsampcurr), 1, batchlen+ntb-nsampcurr);
end

% 2D coordinates for interpolation 
xp = cat(2, rez.xc, rez.yc);

% 2D kernel of the original channel positions 
Kxx = kernel2D(xp, xp, sig);
% 2D kernel of the new channel positions
yp = xp;
yp(:, 2) = yp(:, 2) - shifts;
Kyx = kernel2D(yp, xp, sig);

% kernel prediction matrix
M = Kyx /(Kxx + .01 * eye(size(Kxx,1)));

% the multiplication has to be done on the GPU
dati = gpuArray(single(dat)) * gpuArray(M)';

w_edge = linspace(0, 1, ntb)';
dati(1:ntb, :) = w_edge .* dati(1:ntb, :) + (1 - w_edge) .* dprev;

if size(dati,1)==batchlen+ntb
    dprev = dati(batchlen+[1:ntb], :);
    dati = dati(:, 1:batchlen);
else
    dprev = [];
    dati = dati(:, 1:nsampcurr);
end

dat_cpu = gather(int16(dati));


% we want to write the aligned data back to the same file
fseek(fid, offset, 'bof');
fwrite(fid, dat_cpu', 'int16'); % write this batch to binary file

fclose(fid);

end
