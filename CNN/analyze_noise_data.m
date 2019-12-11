function [R1] = analyze_noise_data(folder_name,ROI,direction_string,camera_bin,binX,binY)

cd(folder_name);
if numel(ROI)==4
 ROI_up=ROI(1); ROI_down=ROI(2);
 ROI_left=ROI(3); ROI_right=ROI(4);   
else
 ROI_up=120+0;
 ROI_down=980-0;
 ROI_left=180+0;
 ROI_right=1120-0;
end

R1.ImagingDirection='vertical';
R1.ImagingDirection='horizontal';
R1.ImagingDirection=direction_string;
 
warning off all
style_name='4x3';
style=hgexport('readstyle',style_name);
M=6E-3;
Na=6.02E23;
m=M/Na;
hbar=1.054571628E-34;
h=hbar*2*pi;
k_b=1.38E-23;
if strcmp(R1.ImagingDirection,'vertical')==1
i_sat = 2.54; %2.54 for vertical imaging, 2.54*2 for horizontal
wrong_polar = 1;%1 for vertical imaging, 1/2 for horizontal
magnification=3.89;
bckg_count = 20.4;
c_gain = 3.8;%3.8 for normal gain and 1.9 for high gain
c_eff = 0.22;
else
i_sat = 2.54*2; %2.54 for vertical imaging, 2.54*2 for horizontal
wrong_polar = 1/2;%1 for vertical imaging, 1/2 for horizontal
magnification=4.91;
bckg_count=200.4;
c_Rout=6.28;
c_gain = 1;%3.8 for normal gain and 1.9 for high gain
c_eff = 0.4;
end
pixel_sizeX=camera_bin*binX*6.45/magnification;
%pixel_sizeX=6.45/magnification;
pixel_sizeY=6.45*binY/magnification;
R1.pixel_sizeX=pixel_sizeX*1E-6;
R1.pixel_sizeY=pixel_sizeY*1E-6;
pulse_width = 5e-6;
c_light = 299792458;
lambda=671E-9;
k0=2*pi/lambda;
detuning=0; % in MHz - laser shift frequency
delta=detuning/5.87;
n_i=3*lambda/2/k0^2;
cross_sec_r=2*k0*n_i*wrong_polar;
R1.cross=cross_sec_r;
R1.gain=c_gain;
R1.Qeff=c_eff;
R1.ReadOut=c_Rout;
Ephoton = 2*pi*hbar*c_light/lambda;
pixel_power = 10*i_sat*pixel_sizeX*pixel_sizeY*1e-12;
pixel_energy = pixel_power*pulse_width;
n_counts = pixel_energy/Ephoton;
i_sat_cal = n_counts*c_eff/c_gain;
beta=1;

fnames = dir(fullfile('*img*.mat'));% find all .mat files
R1.CycleNumber = (numel(fnames));R2.CycleNumber = (numel(fnames));
R1.AtomNumber1 = ones(R1.CycleNumber,1);R2.AtomNumber1 = R1.AtomNumber1;
R1.SignalFileName=num2cell(R1.AtomNumber1);
R1.ReferenceFileName=R1.SignalFileName;
R1.MatName=R1.SignalFileName;
R1.XProfile1=ones(R1.CycleNumber,(ROI_down-ROI_up+1)/binY);R2.XProfile1=R1.XProfile1;
R1.X1=ones(R1.CycleNumber,(ROI_down-ROI_up+1)/binY);R2.X1=R1.X1;
R1.AxialGaussianSize=R1.AtomNumber1;R2.AxialGaussianSize=R1.AxialGaussianSize;
R1.ZProfile1=ones(R1.CycleNumber,(ROI_right-ROI_left+1)/binX);R2.ZProfile1=R1.ZProfile1;
R1.ZProfile1B=R1.ZProfile1;R2.ZProfile1B=R2.ZProfile1;
R1.Z1=ones(R1.CycleNumber,(ROI_right-ROI_left+1)/binX);R2.Z1=R1.Z1;
R1.ZCenter=R1.AtomNumber1;R2.ZCenter=R1.ZCenter;
R1.Power_ref=R1.AtomNumber1;R1.Power_sig=R1.AtomNumber1;
R1.CycleNumber=0;
R1.Correction=R1.AtomNumber1;R2.Correction=R1.Correction;
R1.ro=ones(R1.CycleNumber,(ROI_right-ROI_left+1)*(ROI_down-ROI_up+1)/binY/binX);
R1.OD=R1.ro;R1.Nph_ref=R1.OD;R1.Nph_sig=R1.OD;R1.ODRes=R1.OD;
i=1;
k=1;
while i<=numel(fnames)
R1.CycleNumber=R1.CycleNumber+1;
fname_sigref=fnames(i).name;
data_file=load(fname_sigref);
if isfield(data_file,'Raw_Image')
    data_file.image=data_file.Raw_Image;
end
R1.MatName{k}=fname_sigref;
R1.SignalFileName{k}=data_file.image.signame;
R1.ReferenceFileName{k}=data_file.image.refname;
R1.RFrequency(k)=data_file.image.RFfrq;
R1.CurrentSensor(k,:)=data_file.image.currentsensor;
sig1=data_file.image.sig1matrix-bckg_count;
ref1=data_file.image.ref1matrix-bckg_count;
sig2=data_file.image.sig2matrix-bckg_count;
ref2=data_file.image.ref2matrix-bckg_count;
%absor=(1-sig./ref)*100;figure;imagesc(absor);colorbar;
roi_down=ROI_up-50;
roi_up=roi_down-100;
roi_left=ROI_left;
roi_right=ROI_right;
s_roi=sig1(roi_up:roi_down,roi_left:roi_right);
s_roi=sum(s_roi(:));                 
r_roi=ref1(roi_up:roi_down,roi_left:roi_right);
r_roi=sum(r_roi(:));
corr1=s_roi/r_roi;
R1.Power_ref(R1.CycleNumber)=r_roi;
R1.Power_sig(R1.CycleNumber)=s_roi;
ref1=ref1*corr1; 
R1.Correction(i)=corr1;
s_roi=sig2(roi_up:roi_down,roi_left:roi_right);
s_roi=sum(s_roi(:));                 
r_roi=ref2(roi_up:roi_down,roi_left:roi_right);
r_roi=sum(r_roi(:));
corr2=s_roi/r_roi;
ref2=ref2*corr2; 
R2.Correction(i)=corr2;
if k==1
%if R.RFrequency(k)==12000
absor=(1-sig1(ROI_up:ROI_down,ROI_left:ROI_right)./ref1(ROI_up:ROI_down,ROI_left:ROI_right))*100;
figure('Position',[100 100 900 400]);
h1_abs = axes('Units','normalized','Position',[0.05,0.1,0.44,0.8]);
imagesc(absor);colorbar;
h1_ax = axes('Units','normalized','Position',[0.54,0.1,0.44,0.8]);
plot(sum(absor)/max(sum(absor)));legend('axial');
title(['Profiles at ',num2str(R1.RFrequency(k)/1E3),' kHz']);
hold on;plot(sum(absor,2)/max(sum(absor,2)),'.');hold off
absor=(1-sig2(ROI_up:ROI_down,ROI_left:ROI_right)./ref2(ROI_up:ROI_down,ROI_left:ROI_right))*100;
% figure('Position',[100 100 900 400]);
% h1_abs = axes('Units','normalized','Position',[0.05,0.1,0.44,0.8]);
% imagesc(absor);colorbar;
% h1_ax = axes('Units','normalized','Position',[0.54,0.1,0.44,0.8]);
% plot(sum(absor)/max(sum(absor)));legend('axial');
% title(['Profiles at ',num2str(R1.RFrequency(k)/1E3),' kHz']);
% hold on;plot(sum(absor,2)/max(sum(absor,2)),'.');hold off
%disp(['RF = ',num2str(R.RFrequency(k)/1E3),'kHz']);
waitforbuttonpress();
end
analyzeit1(sig1,ref1);
%analyzeit2(sig2,ref2);
i=i+1;
k=k+1;
%disp([num2str(i),'/',num2str(numel(fnames))]);
end
function analyzeit1(sig_local,ref_local)
    
ref_roi0=ref_local(ROI_up:ROI_down,ROI_left:ROI_right);
ref_roi=sepblockfun(ref_roi0,[binY,binX],'sum');%bin
image_roi0=sig_local(ROI_up:ROI_down,ROI_left:ROI_right);
image_roi=sepblockfun(image_roi0,[binY,binX],'sum');%bin
%analysis
%phase_factor = (1+4*delta^2)*log(beta*ref_roi./(image_roi-(1-beta)*ref_roi))+(ref_roi-image_roi)/i_sat_cal;
%phase_factor = log(beta*ref_roi./(image_roi-(1-beta)*ref_roi))+(ref_roi-image_roi)/i_sat_cal;
phase_factor = (1+4*delta^2)*log(beta*ref_roi./(image_roi-(1-beta)*ref_roi));
ro_zy=phase_factor/cross_sec_r;
R1.OD(R1.CycleNumber,:)=reshape(phase_factor,1,[]);
R1.ro(R1.CycleNumber,:)=reshape(ro_zy,1,[])*1e-12;
R1.AtomNumber1(R1.CycleNumber)=sum(ro_zy(:))*pixel_sizeX*pixel_sizeY*1e-12;
%radial fits
frame_size=size(image_roi);
x_r=1:1:frame_size(1);
y_r=sum(ro_zy')*pixel_sizeX*1e-12;
y_r_n=y_r/max(y_r);
c_r=find(y_r_n==max(y_r_n));
%Gaussian
beta0_r=[min(y_r_n),50,1,c_r];
gamma_r=nlinfit(x_r,y_r_n,@gauss_f1,beta0_r);
R1.RadialGaussianSize(R1.CycleNumber)=gamma_r(2)*pixel_sizeY;
R1.RadialPosition(R1.CycleNumber)=gamma_r(4)*pixel_sizeY;
R1.XProfile1(R1.CycleNumber,:) = y_r - max(y_r)*gamma_r(1);
R1.X1=x_r*pixel_sizeY;
%TF
beta0_r=[min(y_r_n),50,1,c_r];
gamma_r=nlinfit(x_r,y_r_n,@TF1D_f1,beta0_r);
R1.RadialTFSize(R1.CycleNumber)=gamma_r(2)*pixel_sizeY;
%axial fits
x_ax=1:1:frame_size(2);
y_ax=sum(ro_zy)*pixel_sizeY*1e-12;
y_ax_n=y_ax/max(y_ax);
c_ax=find(y_ax_n==max(y_ax_n));
%Gaussian
beta0_ax=[min(y_ax_n),50,1,c_ax];
gamma_ax=nlinfit(x_ax,y_ax_n,@gauss_f1,beta0_ax);
R1.ZProfile1B(R1.CycleNumber,:) = y_ax - max(y_ax)*gamma_ax(1);
R1.ZProfile1(R1.CycleNumber,:) = y_ax;
R1.AxialGaussianSize(R1.CycleNumber)=abs(gamma_ax(2)*pixel_sizeX);
R1.AxialPosition(R1.CycleNumber)=gamma_ax(4)*pixel_sizeX;
R1.Z1=x_ax*pixel_sizeX;
%% 2D Gaussian fit of optical density
beta20=[min(phase_factor(:)) max(phase_factor(:)) R1.AxialGaussianSize(R1.CycleNumber)/pixel_sizeX ...
    R1.RadialGaussianSize(R1.CycleNumber)/pixel_sizeY  round(R1.AxialPosition(R1.CycleNumber)/pixel_sizeX) ...
    round(R1.RadialPosition(R1.CycleNumber)/pixel_sizeY)];
[X,Y]=meshgrid(1:1:frame_size(2),1:1:frame_size(1));
XY=[X;Y];
G2_fit=@(beta20,XY) reshape(gauss_f2D(beta20,XY),1,[]);
gamma_2=nlinfit(XY,R1.OD(R1.CycleNumber,:),G2_fit,beta20);
R1.ODRes(R1.CycleNumber,:)=R1.OD(R1.CycleNumber,:)-G2_fit(gamma_2,XY);
gamma_2(3)=gamma_2(3)*pixel_sizeX;
gamma_2(4)=gamma_2(4)*pixel_sizeY;
R1.G2Dfit(R1.CycleNumber,:)=gamma_2;
%% Photon shot noise
R1.NCount_ref(R1.CycleNumber,:)=reshape(ref_roi,1,[]);
R1.NCount_sig(R1.CycleNumber,:)=reshape(image_roi,1,[]);
end
%%
function analyzeit2(sig_local,ref_local)
    
ref_roi=ref_local(ROI_up:ROI_down,ROI_left:ROI_right);
image_roi=sig_local(ROI_up:ROI_down,ROI_left:ROI_right);
%analysis
phase_factor = (1+4*delta^2)*log(beta*ref_roi./(image_roi-(1-beta)*ref_roi))+(ref_roi-image_roi)/i_sat_cal;
%phase_factor = log(beta*ref_roi./(image_roi-(1-beta)*ref_roi))+(ref_roi-image_roi)/i_sat_cal;
ro_zy=phase_factor/cross_sec_r;
R2.ro(R1.CycleNumber,:)=reshape(ro_zy,1,[])*1e-12;
R2.AtomNumber1(R1.CycleNumber)=sum(ro_zy(:))*pixel_sizeX*pixel_sizeY*1e-12;
%radial fits
frame_size=size(image_roi);
x_r=1:1:frame_size(1);
y_r=sum(ro_zy')*pixel_sizeX*1e-12;
y_r_n=y_r/max(y_r);
c_r=find(y_r_n==max(y_r_n));
%Gaussian
beta0_r=[min(y_r_n),15,1,c_r];
gamma_r=nlinfit(x_r,y_r_n,@gauss_f1,beta0_r);
R2.RadialGaussianSize(R1.CycleNumber)=gamma_r(2)*pixel_sizeY;
R2.RadialPosition(R1.CycleNumber)=gamma_r(4)*pixel_sizeY;
R2.XProfile1(R1.CycleNumber,:) = y_r - max(y_r)*gamma_r(1);
%axial fits
x_ax=1:1:frame_size(2);
y_ax=sum(ro_zy)*pixel_sizeY*1e-12;
y_ax_n=y_ax/max(y_ax);
c_ax=find(y_ax_n==max(y_ax_n));
%Gaussian
beta0_ax=[min(y_ax_n),50,1,c_ax];
gamma_ax=nlinfit(x_ax,y_ax_n,@gauss_f1,beta0_ax);
R2.ZProfile1B(R1.CycleNumber,:) = y_ax - max(y_ax)*gamma_ax(1);
R2.ZProfile1(R1.CycleNumber,:) = y_ax;
R2.AxialGaussianSize(R1.CycleNumber)=gamma_ax(2)*pixel_sizeX;
R2.AxialPosition(R1.CycleNumber)=gamma_ax(4)*pixel_sizeX;
R2.Z1=x_ax*pixel_sizeX;
end

end