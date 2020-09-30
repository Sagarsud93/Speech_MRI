%% Source code for compressed sensing and parallel imaging reconstruction for 2D dynamic MRI of human vocal tract
% Implemented by non-linear CG optimization with L1 temporal TV regularization 

clear all; close all; clc;
addpath('./nufft_toolbox');
addpath('./utilities');

dir_matdata      = './';
name_matdata = 'lac09032017_21_23_13.mat';
dir_resultdata   = './';
fn                      = fullfile(dir_matdata, name_matdata); 
data                  = load (fn);

%% Data Preparation
fprintf('Data preparation... \n')

% Optional: Crop data to first 5 seconds solely for fast computational purposes. 
% However, note the length of full data is typically 15-30 seconds; and also note TR=6ms. 
%
% ntviews_crop = 830;  
% data.kdata = data.kdata(:,:,1:ntviews_crop);
% data.kloc   = data.kloc(:,1:ntviews_crop);

data.kdata  =  permute(data.kdata,[1 3 2]);

param                = struct;
param.nx           =  size(data.kdata,1); % number of samples per a spiral arm
param.ntviews   =  size(data.kdata,2); % number of spiral arms (views)
param.nc           =  size(data.kdata,3); % number of coil channel elements
param.imsize     =  [84 84];  % Image matrix size
param.narms     =  2; % number of desired spiral arms per image time frame        
param.narmsfull = 7; % number of spiral arms for satisfying Nyquist sampling rate
param.nt            =  floor(param.ntviews/param.narms); % number of image time frames

% crop the data according to the number of arms per frame
data.kdata  =  data.kdata(:,1:param.nt*param.narms,:);
data.k         =  data.kloc(:,1:param.nt*param.narms);
data.w        =  repmat(data.w(:,1), [1 param.nt*param.narms]);% density pre-compensation weight

data.kdata  =  1.4790e3*data.kdata./max(abs(data.kdata(:))); % scaling
data            =  rmfield(data,'kloc');

fprintf('Data preparation...DONE \n')

%% Coil Map Estimation
fprintf('Coil map estimation... \n')

param.kernelsize_walsh = 20;
csm  =  est_coilmaps(data, param);

fprintf('Coil map estimation...DONE \n')

csm_plot = save_3d_static_img (csm, 90, param.nc, 1);

figure,  
fig=subplot(2,1,1); imagesc( abs(csm_plot)); colormap(fig,'gray'); axis image off; title('Magnitude of CSM');
fig=subplot(2,1,2); imagesc( angle(csm_plot));  colormap(fig,'jet');axis image off; title('Phase of CSM');

%% CS Reconstruction
fprintf('Preparation for CS reconstruction... \n') 

% rearrange the data into a time-series
kdatau        =  zeros(param.nx, param.narms, param.nc, param.nt );
ku               =  zeros(param.nx, param.narms, param.nt );
wu              =  zeros(param.nx, param.narms, param.nt );
for ii=1:param.nt     
    views = (ii-1)*param.narms+1:ii*param.narms;    
    kdatau(:,:,:,ii) = data.kdata(:,views,:); % undersampled data
    ku(:,:,ii)          = data.k(:,views);
    wu(:,:,ii)         = data.w(:,views);
    
end

param.E           = MCNUFFT(ku, wu, csm);% multicoil NUFFT operator
param.y           = kdatau.*permute(repmat(sqrt(wu),[1 1 1 param.nc]),[1 2 4 3]); 

recon_nufft      = param.E'*param.y; % Zero-filled reconstruction (initial guess)

param.W          = TV_Temp(); % temporal TV regularization
%param.lambda = [0.2]*max(abs(recon_nufft(:))); % the regularization parameter - set to 0.2; 
param.lambda = [0.0]*max(abs(recon_nufft(:))); % the regularization parameter - set to 0.2;
param.nite       = 8;
param.niteOut = 4;
param.display  =1;

fprintf('Preparation for CS reconstruction...DONE \n')

fprintf('\n Start CG iterations... \n')
tic;
recon_cs = recon_nufft;
for n=1:param.niteOut,
    [recon_cs, cost, l2cost, l1cost,finalcost] = CSL1NlCg(recon_cs,param);
end
param.recon_time = toc
fprintf('\n End CG iterations... \n')
%% Save Results
mkdir(dir_resultdata); cd(dir_resultdata);
name_outfile = fullfile(dir_resultdata, strcat(name_matdata(1:end-4),'_result'));

save(name_outfile, 'recon_cs', 'param');
save_video_speech2D(strcat(name_outfile,'.avi'), recon_cs, 90, 1/(param.narms*6.004e-3));

