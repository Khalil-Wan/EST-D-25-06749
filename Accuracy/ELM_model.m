clc
close all
clear all;
rng(0)

EIS_data=load('EIS_data.txt');
P_train = EIS_data(:,[1:124])';
T_train = EIS_data(:,125)'; % SOH of training set
M = size(P_train, 2);

EIS_data_35C02=load('EIS_data_35C02.txt');
P_test  = EIS_data_35C02(:,[1:124])';
T_test  = EIS_data_35C02(:,125)';% SOH of testing set
N = size(P_test, 2);

%  Data normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  Transpose to fit the model
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

pp_train= p_train(:,[1:121]);%  change the input features
pp_test = p_test(:,[1:121]);%  change the input features

%  Create the model
num_hiddens = 300;        % number of the hiddens
activate_model = 'sig';   % activation function
[IW, B, LW, TF, TYPE] = elmtrain(pp_train', t_train', num_hiddens, activate_model, 0);

%  Simulation test
t_sim1 = elmpredict(pp_train', IW, B, LW, TF, TYPE);
t_sim2 = elmpredict(pp_test' , IW, B, LW, TF, TYPE);

%  Inverse normalization of data
T_sim1 =  t_sim1';
T_sim2 =  t_sim2';
mm=t_test./t_test(1,1);
nn=T_sim2./T_sim2(1,1);

%  Draw a figure
figure(1);
% f = [T_sim2+sqrt(t_sim2_var); flipdim(T_sim2-sqrt(t_sim2_var),1)]; 
% h=fill([[2:2:2*length(T_sim2)]'; flipdim([2:2:2*length(T_sim2)]',1)], f, [255 191 200]/255);
% set(h,'LineStyle','none');set(gcf,'color','w');
hold on; 
plot([2:2:2*length(t_test)],t_test,'x','color', [0 130 216]/255,'LineWidth',3);
plot([2:2:2*length(nn)],nn,'+','color',[205 39 70]/255,'LineWidth',3);
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