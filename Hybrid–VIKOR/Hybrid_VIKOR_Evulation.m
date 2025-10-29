clc
clear all
close all

data  =load('data.txt');
% Data Normalization 
[m, n] = size(data);
for j = 1:n
    data(:, j) = (max(data(:, j)) - data(:, j)) / (max(data(:, j))-min(data(:, j)));
end

% Entropy method
prob = data./ repmat(sum(data, 1), m, 1);
k=1/log(m);
entropy = 1+k*sum(prob.* log(prob + eps), 1);
sum_entropy = sum(entropy);
w1 =  entropy./ sum_entropy;

% Critic method
std_dev = std(data);
corr_matrix = corrcoef(data);
R=sum(1 - corr_matrix, 1);
C = std_dev.* R;
sum_C = sum(C);
w2 = C./ sum_C;

% Std
total_std = sum(std_dev);
w3 = std_dev / total_std;

alpha = 1/3;
beta = 1/3;
gamma = 1/3;
w = alpha * w1+beta * w2 + gamma * w3;% weight

f_star = min(data,[],1);
f_double_star = max(data,[],1);

% Caculate S
S = zeros(m, 1);
for i = 1:m
    for j = 1:n
        S(i)=S(i)+w(j)*((f_double_star(j)-data(i,j))/(f_double_star(j)-f_star(j)));
    end
end

% Caculate RR
RR = zeros(m, 1);
for i = 1:m
    temp = zeros(n, 1);
    for j = 1:n
        temp(j)=w(j)*((f_double_star(j)-data(i,j))/(f_double_star(j)-f_star(j)));
    end
    RR(i)=max(temp);
end

% Calculate minimum and maximum values of S and R
S_star = min(S);
S_double_star = max(S);
R_star = min(RR);
R_double_star = max(RR);

% Assuming v = 0.5
v = 0.5;
% Caculate Qi
Q = v*((S - S_star)/(S_double_star - S_star))+(1 - v)*((RR - R_star)/(R_double_star - R_star));
disp("complete");
bar(Q)
[paixu,I] = sort(Q);