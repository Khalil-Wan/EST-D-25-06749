clc
close all
clear all;
rng(0)

EIS_data=load('EIS_data.txt');
mean1 = mean(EIS_data(:,[1:124]),1); std1 = std(EIS_data(:,[1:124]),1);
X_train_0 = zscore (EIS_data(:,[1:124]));
X_train   = X_train_0(:,[1:121]);% change the input features
Y_train = EIS_data(:,125);% SOH of training set

meanfunc = @meanZero; hyp.mean = [];                 % mean function is zero
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);    % the Gaussian likelihood
covfunc = @covSEiso; hyp.cov = log([0.9;2]);         % Squared Exponential covariance function
hyp_EIS_Capacity = minimize(hyp, @gp, -10000, @infGaussLik, meanfunc, covfunc, likfunc, X_train, Y_train);

EIS_data_35C02=load('EIS_data_35C02.txt');
X_test_35C02_0 = (EIS_data_35C02(:,[1:124])-mean1)./std1;
X_test_35C02   =  X_test_35C02_0(:,[1:121]);% change the input features
capacity35C02=EIS_data_35C02(:,125);% SOH of testing set
[Y_test_cap_35C02,Y_test_cap_35C02_var] = gp(hyp_EIS_Capacity,@infGaussLik,meanfunc,covfunc,likfunc,X_train, Y_train, X_test_35C02);

mm=capacity35C02/capacity35C02(1,1);
nn=Y_test_cap_35C02/Y_test_cap_35C02(1,1);

% Draw a figure
figure(1);
f = [Y_test_cap_35C02/Y_test_cap_35C02(1,1)+sqrt(Y_test_cap_35C02_var)/Y_test_cap_35C02(1,1); flipdim(Y_test_cap_35C02/Y_test_cap_35C02(1,1)-sqrt(Y_test_cap_35C02_var)/Y_test_cap_35C02(1,1),1)]; 
h=fill([[2:2:2*length(Y_test_cap_35C02)]'; flipdim([2:2:2*length(Y_test_cap_35C02)]',1)], f, [255 191 200]/255);
set(h,'LineStyle','none');set(gcf,'color','w');
hold on; 
plot([2:2:2*length(Y_test_cap_35C02)],capacity35C02/capacity35C02(1,1),'x','color', [0 130 216]/255,'LineWidth',3);
plot([2:2:2*length(Y_test_cap_35C02)],Y_test_cap_35C02/Y_test_cap_35C02(1,1),'+','color',[205 39 70]/255,'LineWidth',3);
% xlim([0 2*w]);
% ylim([0.6 1.04]); 
% xlabel('\fontsize{25}Cycle Number');
% ylabel('\fontsize{25}Identified Capacity');
% title ('\fontsize{25}35C02');
lgd = legend({'','\fontsize{20}Measured','\fontsize{20}Estimated'},'Box','off');

% Calculat the indexes
NN=length(mm);
R2=1-norm(mm-nn)^2 / norm(mm-mean(mm))^2;
WIA=calculateWIA(mm,nn);
MSE=sum((mm-nn).^2)./NN;
RMSE=sqrt(sum((mm-nn).^2)./NN);
MAPE=sum(abs((mm-nn)./mm))./NN;
MAE=sum(abs(mm-nn))./NN;

% Output the indexes
ZZ1=[1-R2,1-WIA,MSE,RMSE,MAPE,MAE];
