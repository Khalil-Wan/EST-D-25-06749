clc
close all
clear all;
rng(0)

EIS_data=load('EIS_data.txt');
mean1 = mean(EIS_data(:,[1:124]),1); std1 = std(EIS_data(:,[1:124]),1);
p_train_0 = zscore (EIS_data(:,[1:124]));
p_train   = p_train_0(:,[1:121]);% change the input features
t_train = EIS_data(:,125);% SOH of training set

EIS_data_35C02=load('EIS_data_35C02.txt');
p_test_0 = (EIS_data_35C02(:,[1:124])-mean1)./std1;

[m_noise, n_noise]=size(p_test_0);
% Add different noise to different features, take Level III as an example.
Data_noise(:,1)  =p_test_0(:,1)+0.03*randn(m_noise,1);% Temperature
Data_noise(:,122)=p_test_0(:,122)+0.05*randn(m_noise,1);% CT
Data_noise(:,123)=p_test_0(:,123)+0.05*randn(m_noise,1);% DT
Data_noise(:,124)=p_test_0(:,124)+0.05*randn(m_noise,1);% CE
% Frequence
frequencies = [20004.453	15829.126	12516.703	9909.4424	7835.48	6217.2461	4905.291	3881.2737	3070.9827	2430.7778	1923.1537	1522.4358	1203.8446	952.86591	754.27557	596.71857	471.96338	373.20856	295.47278	233.87738	185.05922	146.35823	115.77804	91.6721	72.51701	57.36816	45.3629	35.93134	28.40909	22.48202	17.79613	14.06813	11.1448	8.81772	6.97545	5.5173	4.36941	3.45686	2.73547	2.16054	1.70952	1.35352	1.07079	0.84734	0.67072	0.53067	0.41976	0.33183	0.26261	0.20791	0.16452	0.13007	0.10309	0.08153	0.06443	0.05102	0.04042	0.03192	0.02528	0.01999];
% level of noise
noise_level = 0.015; % level III
X_eis_original = p_test_0(:, 2:121); 
X_eis_noisy = add_eis_noise_with_level(X_eis_original, frequencies, noise_level);
Data_noise(:, 2:121) = X_eis_noisy(:, 1:120);% EIS

p_test  =  Data_noise(:,[1:121]);
t_test  =  EIS_data_35C02(:,125);

%  Create the model 
trees = 100;                                      
leaf  = 5;                                        
OOBPrediction = 'on';                             
OOBPredictorImportance = 'on';                    
Method = 'regression';                            
net = TreeBagger(trees, p_train, t_train, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  

%  Simulation test
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test);

%  Inverse normalization of data
T_sim1 =  t_sim1;
T_sim2 =  t_sim2;
mm=t_test;
nn=T_sim2;

% Draw a figure
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

function X_noisy = add_eis_noise_with_level(X_original, frequencies, noise_level)
       
    [m_noise, n_noise] = size(X_original);
    X_noisy = X_original;
    
    rho_low = 0.401;   % low frequence (0-1Hz)
    rho_mid = 0.588;   % middle frequence (1-100Hz)
    rho_high = 0.450;  % high frequence (>100Hz)
    
    for i = 1:m_noise 
        for j = 1:60 
            freq = frequencies(j);
            
            real_idx = j + 0;   % real
            imag_idx = j + 60;  % imaginary
            
            Z_real = X_original(i, real_idx);
            Z_imag = X_original(i, imag_idx);
            Z_mag = sqrt(Z_real^2 + Z_imag^2);
            
            if freq < 1   % low frequence
                freq_factor = 1/sqrt(freq);
                rho = rho_low;
            elseif freq < 100 % middle frequence
                freq_factor = 1/freq;
                rho = rho_mid;
            else % high frequence
                freq_factor = 1;
                rho = rho_high;
            end
            
            % Calculate noise standard deviation 
            sigma_base = noise_level * Z_mag * freq_factor;
            
            % noise
            covariance_matrix = sigma_base^2 * [1, rho; rho, 1];
            noise = mvnrnd([0, 0], covariance_matrix, 1);
            
            % add noise
            X_noisy(i, real_idx) = Z_real + noise(1);
            X_noisy(i, imag_idx) = Z_imag + noise(2);
        end
    end
end
