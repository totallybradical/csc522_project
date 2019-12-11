%% Feb 14, 2019
%% rfdata_Cold
close all; clear all
kb=1.38E-23;hbar=1.054571628E-34;h=hbar*2*pi;
M=6.015E-3;Na=6.02E23;m=M/Na;
nu_X=23;nu_R=625;
% t=0
folder_name='C:\Users\ilya\North Carolina State University\CSS522 - Documents\Noise Data\February14_2019\rfdata';
ROI_up=580-1; ROI_down=610; ROI_left=121; ROI_right=480;
%ROI_up=580-1-100; ROI_down=610+100; ROI_left=121; ROI_right=480;
xbin=4;ybin=2;
%xbin=1;ybin=1;
clear R1 R2;
[R1]=analyze_noise_data(folder_name,[ROI_up ROI_down ROI_left ROI_right],'horizontal',2,xbin,ybin);
hor_frame=(ROI_right-ROI_left+1)/xbin;
ver_frame=(ROI_down-ROI_up+1)/ybin;
p_area=R1.pixel_sizeX*R1.pixel_sizeY;
fname2save='C:\Users\ilya\North Carolina State University\CSS522 - Documents\project_data1\cold_train\';
fname2save='C:\Users\ilya\North Carolina State University\CSS522 - Documents\project_data1\cold\';
for i=1:numel(R1.AtomNumber1)
    OD_matrix=reshape(R1.OD(i,:),ver_frame,[])*p_area/R1.cross;
    %writematrix(OD_matrix,[fname2save,'c_train_',num2str(i),'.csv']);
    imwrite(OD_matrix,[fname2save,'c_train_',num2str(i),'.tiff']);
end
mean(R1.AtomNumber1)
N1=trapz(R1.ZProfile1,2)*R1.pixel_sizeX;
mean(N1)*1E6
N1B=trapz(R1.ZProfile1B,2)*R1.pixel_sizeX;
mean(N1B)*1E6

% mean OD
OD_mean=reshape(mean(R1.OD),ver_frame,[]);
%figure;imagesc(OD_mean);colorbar
varOD=reshape((std(R1.OD)).^2,ver_frame,[]);
ReadOut=R1.ReadOut;
%figure;imagesc(varOD*(p_area/R1.cross)^2);colorbar
Snoise_sig=R1.gain./reshape(mean(R1.NCount_sig),ver_frame,[]);
Snoise_ref=R1.gain./reshape(mean(R1.NCount_ref),ver_frame,[]);
Rnoise_sig=(ReadOut*R1.gain./reshape(mean(R1.NCount_sig),ver_frame,[])).^2;
Rnoise_ref=(ReadOut*R1.gain./reshape(mean(R1.NCount_ref),ver_frame,[])).^2;
%figure;imagesc(Snoise_sig);colorbar
%figure;imagesc(Snoise_ref);colorbar
%figure;imagesc(Rnoise_sig);colorbar
%figure;imagesc(Rnoise_ref);colorbar
varNa=(varOD-Snoise_sig-Snoise_ref-Rnoise_sig-Rnoise_ref)*(p_area/R1.cross)^2;
figure;imagesc(varNa);colorbar();
Na_mean=OD_mean*(p_area/R1.cross);
figure;imagesc(Na_mean);colorbar();
sum(Na_mean(:))
varNa=varNa-21;
figure;plot(Na_mean(:),varNa(:),'.');
%%
y_cut=4:15;x_cut=25:72;
figure;plot(reshape(Na_mean(y_cut,x_cut),1,[]),reshape(varNa(y_cut,x_cut),1,[]),'.');
R1.Class=R1.MatName;
for i=1:numel(R1.AtomNumber1)
    R1.Class{i}='COLD';
end
file_name='C:\Users\ilya\OneDrive\Documents\projects\NCSU_graduate\CSC522\Project\data\Feb14\Feb14_COLD_metadata.csv';
save_structure(R1,file_name);
cold_out=array2table([(reshape(Na_mean(y_cut,x_cut),1,[]))',(reshape(varNa(y_cut,x_cut),1,[])')]);
cold_out.Properties.VariableNames = {'Mean atom number','Variance'};
writetable(cold_out,'E:\Clouds\OneDrive\Documents\projects\NCSU_graduate\CSC522\Project\data\Feb14\Feb14_COLD.csv','Delimiter',',');

%% Hot
%% rfdata_Hot
%close all; clear all
kb=1.38E-23;hbar=1.054571628E-34;h=hbar*2*pi;
M=6.015E-3;Na=6.02E23;m=M/Na;
nu_X=23;nu_R=625;
% t=0
folder_name='C:\Users\ilya\North Carolina State University\CSS522 - Documents\Noise Data\February14_2019\rfdata2';
ROI_up=580-1; ROI_down=610; ROI_left=121; ROI_right=480;
xbin=4;
ybin=2;
%xbin=6;
%ybin=3;
clear R1 R2;
[R1]=analyze_noise_data(folder_name,[ROI_up ROI_down ROI_left ROI_right],'horizontal',2,xbin,ybin);
hor_frame=(ROI_right-ROI_left+1)/xbin;
ver_frame=(ROI_down-ROI_up+1)/ybin;
p_area=R1.pixel_sizeX*R1.pixel_sizeY;
fname2save='C:\Users\ilya\North Carolina State University\CSS522 - Documents\project_data\hot_train\';
fname2save='C:\Users\ilya\North Carolina State University\CSS522 - Documents\project_data1\hot\';
for i=1:numel(R1.AtomNumber1)
    OD_matrix=reshape(R1.OD(i,:),ver_frame,[])*p_area/R1.cross;
    %writematrix(OD_matrix,[fname2save,'h_train_',num2str(i),'.csv']);
    imwrite(OD_matrix,[fname2save,'h_train_',num2str(i),'.jpeg']);
end
mean(R1.AtomNumber1)
N1=trapz(R1.ZProfile1,2)*R1.pixel_sizeX;
mean(N1)*1E6
N1B=trapz(R1.ZProfile1B,2)*R1.pixel_sizeX;
mean(N1B)*1E6
% mean OD
OD_mean=reshape(mean(R1.OD),ver_frame,[]);
%figure;imagesc(OD_mean);colorbar
varOD=reshape((std(R1.OD)).^2,ver_frame,[]);
%varOD=reshape(R1.ODRes,ver_frame,[]);
ReadOut=R1.ReadOut;
%figure;imagesc(varOD);colorbar
Snoise_sig=R1.gain./reshape(mean(R1.NCount_sig),ver_frame,[]);
Snoise_ref=R1.gain./reshape(mean(R1.NCount_ref),ver_frame,[]);
Rnoise_sig=(ReadOut*R1.gain./reshape(mean(R1.NCount_sig),ver_frame,[])).^2;
Rnoise_ref=(ReadOut*R1.gain./reshape(mean(R1.NCount_ref),ver_frame,[])).^2;
%figure;imagesc(Snoise_sig);colorbar
%figure;imagesc(Snoise_ref);colorbar
%figure;imagesc(Rnoise_sig);colorbar
%figure;imagesc(Rnoise_ref);colorbar
varNa=(varOD-Snoise_sig-Snoise_ref-Rnoise_sig-Rnoise_ref)*(p_area/R1.cross)^2;
figure;imagesc(varNa);colorbar();
Na_mean=OD_mean*(p_area/R1.cross);
figure;imagesc(Na_mean);colorbar();
sum(Na_mean(:))
varNa=varNa-22;
figure;plot(Na_mean(:),varNa(:),'.');
%%
y_cut=4:15;x_cut=20:72;
figure;plot(reshape(Na_mean(y_cut,x_cut),1,[]),reshape(varNa(y_cut,x_cut),1,[]),'.');
R1.Class=R1.MatName;
for i=1:numel(R1.AtomNumber1)
    R1.Class{i}='HOT';
end
file_name='E:\Clouds\OneDrive\Documents\projects\NCSU_graduate\CSC522\Project\data\Feb14\Feb14_HOT_metadata.csv';
save_structure(R1,file_name);
cold_out=array2table([(reshape(Na_mean(y_cut,x_cut),1,[]))',(reshape(varNa(y_cut,x_cut),1,[])')]);
cold_out.Properties.VariableNames = {'Mean atom number','Variance'};
writetable(cold_out,'E:\Clouds\OneDrive\Documents\projects\NCSU_graduate\CSC522\Project\data\Feb14\Feb14_HOT.csv');

%% Train data feb 14
fname='C:\Users\ilya\OneDrive\Documents\projects\NCSU_graduate\CSC522\Project\midterm\mid_data\';
%fname='E:\Clouds\OneDrive\Documents\projects\NCSU_graduate\CSC522\Project\midterm\mid_data\';
cdata=readtable([fname,'Feb14_COLD_analyzed.csv']);
hdata=readtable([fname,'Feb14_HOT_analyzed.csv']);
figure;plot(cdata.MeanAtomNumber,cdata.Variance,'.');
hold on;plot(hdata.MeanAtomNumber,hdata.Variance,'.');
plot([0 125],[0,125],'--');
ylim([-20 120]);xlim([-5 130]);
%%
c_mean=cdata.MeanAtomNumber;
c_var=cdata.Variance;
h_mean=hdata.MeanAtomNumber;
h_var=hdata.Variance;
figure;plot(c_mean,c_var,'.');hold on;
plot(h_mean,h_var,'.');
[cold2,gof_cold] = fit(c_mean,c_var,'poly2');
%figure;plot(cold2,c_mean,c_var);hold on;
[hot2,gof_hot] = fit(h_mean,h_var,'poly2');
%plot(hot2,h_mean,h_var);
%% test data feb12 cold

fname='C:\Users\ilya\OneDrive\Documents\projects\NCSU_graduate\CSC522\Project\midterm\mid_data\';
%fname='E:\Clouds\OneDrive\Documents\projects\NCSU_graduate\CSC522\Project\midterm\mid_data\';
cdata_test=readtable([fname,'Feb12_COLD_test.csv']);
test_mean=cdata_test.MeanAtomNumber;
test_var=cdata_test.Variance;
%plot test vs fits
figure;plot(test_mean,test_var,'.');hold on;
plot(0:125,cold2(0:125));plot(0:125,hot2(0:125));
ylim([-20 120]);xlim([-5 130]);

cold_pred=cold2(test_mean(test_mean>20));
hot_pred=hot2(test_mean(test_mean>20));
SSR_hot=sum((test_var(test_mean>20)-hot_pred).^2)/numel(hot_pred)
SSR_cold=sum((test_var(test_mean>20)-cold_pred).^2)/numel(hot_pred)
%% partition training data
[cold2_part,gof_cold_part] = fit([c_mean;h_mean],[c_var;h_var],'poly2');
figure;plot(c_mean,c_var,'.');hold on;
plot(h_mean,h_var,'.');
plot(0:125,cold2_part(0:125));
ylim([-20 120]);xlim([-5 130]);
figure;plot(test_mean,test_var,'.');hold on;plot(0:125,cold2_part(0:125));
%difference between bourder and data
test_err=test_var(test_mean>20)-cold2_part(test_mean(test_mean>20));
1-numel(find(test_err<=0))/numel(test_err)
%% nearest neighbors
T10=[[c_mean;h_mean],[c_var;h_var],[ones(numel(c_mean),1);zeros(numel(h_mean),1)]];
T10_test=[test_mean(test_mean>20),test_var(test_mean>20)];
class_test5=-ones(numel(test_mean(test_mean>20)),1);
class_test10=-ones(numel(test_mean(test_mean>20)),1);
for i=1:numel(test_mean(test_mean>20))
    ind5=knnsearch(T10(:,1:2),T10_test(i,:),'K',5,'Distance','euclidean');
    class_test5(i)=mode(T10(ind5,3));
    ind10=knnsearch(T10(:,1:2),T10_test(i,:),'K',10,'Distance','euclidean');
    class_test10(i)=mode(T10(ind10,3));
end
sum(class_test5)/numel(class_test5)
sum(class_test10)/numel(class_test10)

class_testj=-ones(numel(test_mean(test_mean>20)),1);
classK=2:25;
for j=2:25
for i=1:numel(test_mean(test_mean>20))
    indj=knnsearch(T10(:,1:2),T10_test(i,:),'K',j,'Distance','euclidean');
    class_testj(i)=mode(T10(indj,3));
end
    classK(j-1)=sum(class_testj)/numel(class_testj);
end
figure;plot(2:25,classK);xlim([1 26])
xlabel('k-value');ylabel('Accuracy');