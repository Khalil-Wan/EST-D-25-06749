function wia = calculateWIA(observed, predicted)
    if length(observed) ~= length(predicted)
        error('Observed and predicted value vectors must have the same length');
    end
    num = sum((observed - predicted).^2);
    obs_mean = mean(observed);
    den = sum((abs(predicted - obs_mean)+abs(observed - obs_mean)).^2);
    wia = 1 - num/den;
end