clear;clc;close all;

sample = csvread('solar_samples.csv');
labels = csvread('solar_labels.csv');


plot(sample(100, 1:288), 'linewidth', 1.2);

for i = 1 : 12
    class(i).index = find(labels == (i-1));
    class(i).power = sample(class(i).index, :);
end

figure;
for k = 1 : 12
    subplot(3, 4, k)
    for i = 1 : length(class(k).index)
        plot(class(k).power(i, 1:288));
        hold on
    end
    title([num2str(k), 'Month'])
    ylim([0 8])
end

figure;
for k = 1 : 12
    subplot(3, 4, k)
    temp = class(k).power;
    [m, n] = size(temp);
    temp = reshape(temp, m*n, 1);
    cdfplot(temp)
    title([num2str(k), 'Month'])
end

