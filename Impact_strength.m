%% COMPLETE IMPACT STRENGTH PREDICTION WITH VISUALIZATION
% This code trains multiple ML models and generates comprehensive visualizations

clear; clc; close all;

%% 1. LOAD AND PREPARE DATA
fprintf('Loading and preparing data...\n');

% Load the dataset
data = [
    0.15, 30, 10.9, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.15, 50, 12.36, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.15, 70, 13.7, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.2, 30, 12.96, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.2, 50, 13.73, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.2, 70, 16.13, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.25, 30, 14.02, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.25, 50, 15, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.25, 70, 17.04, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.15, 30, 10.5, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.15, 50, 11.26, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.15, 70, 12.13, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.2, 30, 11.38, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.2, 50, 12.46, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.2, 70, 13.55, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.25, 30, 13.86, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.25, 50, 14.96, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.25, 70, 16.4, 1, 1.75, 0.4, 200, 60, 3000, 1;
    0.2, 100, 4.2, 1, 1.75, 0.4, 220, 60, 2400, 2;
    0.2, 75, 2.9, 1, 1.75, 0.4, 220, 60, 2400, 2;
    0.2, 50, 2.42, 1, 1.75, 0.4, 220, 60, 2400, 3;
    0.2, 75, 1.7, 1, 1.75, 0.4, 220, 60, 2400, 3;
    0.2, 100, 1.85, 1, 1.75, 0.4, 220, 60, 2400, 4;
    0.2, 50, 2.58, 1, 1.75, 0.4, 220, 60, 2400, 4;
    0.2, 100, 3.15, 1, 1.75, 0.4, 220, 60, 2400, 5;
    0.2, 50, 2.75, 1, 1.75, 0.4, 220, 60, 2400, 6;
    0.2, 75, 3.54, 1, 1.75, 0.4, 220, 60, 2400, 7;
    0.2, 10, 1.79, 1, 1.75, 0.4, 205, 60, 3600, 1;
    0.2, 30, 2.291, 1, 1.75, 0.4, 205, 60, 3600, 1;
    0.2, 50, 1.921, 1, 1.75, 0.4, 205, 60, 3600, 1;
    0.2, 10, 1.79, 2, 1.75, 0.4, 205, 60, 3600, 1;
    0.2, 30, 1.799, 2, 1.75, 0.4, 205, 60, 3600, 1;
    0.2, 50, 3.571, 2, 1.75, 0.4, 205, 60, 3600, 1;
    0.2, 10, 1.432, 3, 1.75, 0.4, 205, 60, 3600, 1;
    0.2, 30, 2.694, 3, 1.75, 0.4, 205, 60, 3600, 1;
    0.2, 50, 3.249, 3, 1.75, 0.4, 205, 60, 3600, 1;
];

% Extract features and target
X = data(:, [1, 2, 4:10]);  % Features
y = data(:, 3);  % Impact strength (target)

% Split data (80-20)
rng(42);
cv = cvpartition(length(y), 'HoldOut', 0.2);
idx_train = training(cv);
idx_test = test(cv);

X_train = X(idx_train, :);
y_train = y(idx_train, :);
X_test = X(idx_test, :);
y_test = y(idx_test, :);

fprintf('Data loaded: %d training, %d test samples\n', length(y_train), length(y_test));

%% 2. NORMALIZE DATA
fprintf('Normalizing data...\n');
[X_train_norm, mu, sigma] = zscore(X_train);
X_test_norm = (X_test - mu) ./ sigma;
[y_train_norm, mu_y, sigma_y] = zscore(y_train);

%% 3. TRAIN MODELS
fprintf('\nTraining models...\n');

% 3.1 Multi-Layer Perceptron (MLP)
net_mlp = fitnet(10);
net_mlp.trainFcn = 'trainlm';
net_mlp.trainParam.showWindow = false;
[net_mlp, ~] = train(net_mlp, X_train_norm', y_train_norm');
y_pred_norm_mlp = net_mlp(X_test_norm');
y_pred_mlp = y_pred_norm_mlp' * sigma_y + mu_y;
mse_mlp = mean((y_test - y_pred_mlp).^2);
rmse_mlp = sqrt(mse_mlp);
r2_mlp = 1 - sum((y_test - y_pred_mlp).^2) / sum((y_test - mean(y_test)).^2);

% 3.2 Random Forest
rf_model = TreeBagger(100, X_train_norm, y_train_norm, ...
    'Method', 'regression', 'OOBPredictorImportance', 'on', 'MinLeafSize', 5);
y_pred_norm_rf = predict(rf_model, X_test_norm);
y_pred_rf = y_pred_norm_rf * sigma_y + mu_y;
mse_rf = mean((y_test - y_pred_rf).^2);
rmse_rf = sqrt(mse_rf);
r2_rf = 1 - sum((y_test - y_pred_rf).^2) / sum((y_test - mean(y_test)).^2);

% 3.3 Gradient Boosting
gb_model = fitrensemble(X_train_norm, y_train_norm, ...
    'Method', 'LSBoost', 'NumLearningCycles', 100, 'LearnRate', 0.1);
y_pred_norm_gb = predict(gb_model, X_test_norm);
y_pred_gb = y_pred_norm_gb * sigma_y + mu_y;
mse_gb = mean((y_test - y_pred_gb).^2);
rmse_gb = sqrt(mse_gb);
r2_gb = 1 - sum((y_test - y_pred_gb).^2) / sum((y_test - mean(y_test)).^2);

% 3.4 Decision Tree
tree_model = fitrtree(X_train_norm, y_train_norm, 'MinParentSize', 10);
y_pred_norm_tree = predict(tree_model, X_test_norm);
y_pred_tree = y_pred_norm_tree * sigma_y + mu_y;
mse_tree = mean((y_test - y_pred_tree).^2);
rmse_tree = sqrt(mse_tree);
r2_tree = 1 - sum((y_test - y_pred_tree).^2) / sum((y_test - mean(y_test)).^2);

% 3.5 Ensemble (weighted average)
weights = 1 ./ [rmse_mlp, rmse_rf, rmse_gb, rmse_tree];
weights = weights / sum(weights);
y_pred_ensemble = weights(1)*y_pred_mlp + weights(2)*y_pred_rf + ...
                  weights(3)*y_pred_gb + weights(4)*y_pred_tree;
rmse_ensemble = sqrt(mean((y_test - y_pred_ensemble).^2));
r2_ensemble = 1 - sum((y_test - y_pred_ensemble).^2) / sum((y_test - mean(y_test)).^2);

fprintf('Training complete!\n');

%% 4. FIGURE 1: MODEL COMPARISON (6 subplots)
fprintf('\nGenerating Figure 1: Model Comparison...\n');

figure('Position', [50, 50, 1400, 800], 'Name', 'Model Performance Analysis', 'NumberTitle', 'off');

% Colors for models
colors = lines(5);
model_names = {'MLP', 'Random Forest', 'Gradient Boosting', 'Decision Tree', 'Ensemble'};
all_predictions = [y_pred_mlp, y_pred_rf, y_pred_gb, y_pred_tree, y_pred_ensemble];

% Subplot 1: Actual vs Predicted
subplot(2, 3, 1);
hold on;
markers = {'o', 's', '^', 'd', '*'};
for i = 1:5
    scatter(y_test, all_predictions(:, i), 60, colors(i,:), markers{i}, 'filled', ...
        'MarkerFaceAlpha', 0.7, 'DisplayName', model_names{i});
end
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', 'LineWidth', 2, 'DisplayName', 'Ideal');
xlabel('Actual Impact Strength (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Predicted Impact Strength (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
title('Actual vs Predicted Values', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 9);
grid on;
box on;

% Subplot 2: Residuals
subplot(2, 3, 2);
residuals = y_test - all_predictions;
boxplot(residuals, 'Labels', model_names, 'Whisker', 1.5);
ylabel('Residuals (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
title('Residual Distribution', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 10);

% Subplot 3: Performance Metrics
subplot(2, 3, 3);
metrics = [rmse_mlp, rmse_rf, rmse_gb, rmse_tree, rmse_ensemble;
           r2_mlp, r2_rf, r2_gb, r2_tree, r2_ensemble];
bar(1:5, metrics');
set(gca, 'XTick', 1:5);
set(gca, 'XTickLabel', model_names, 'FontSize', 10);
ylabel('Score', 'FontSize', 11, 'FontWeight', 'bold');
title('Performance Metrics Comparison', 'FontSize', 12, 'FontWeight', 'bold');
legend({'RMSE (kJ/m²)', 'R²'}, 'Location', 'northwest', 'FontSize', 9);
grid on;
ylim([0, max([metrics(:); 1]) * 1.1]);

% Subplot 4: Feature Importance
subplot(2, 3, 4);
feature_names = {'LayerThick', 'Infill%', 'FillPattern', 'FilamentD', ...
                 'NozzleD', 'NozzleT', 'BedT', 'PrintSpeed', 'Color'};
importance = rf_model.OOBPermutedPredictorDeltaError;
[importance_sorted, idx_sorted] = sort(importance, 'descend');
barh(importance_sorted, 'FaceColor', [0.2, 0.4, 0.8]);
set(gca, 'YTick', 1:length(feature_names));
set(gca, 'YTickLabel', feature_names(idx_sorted), 'FontSize', 10);
xlabel('Importance Score', 'FontSize', 11, 'FontWeight', 'bold');
title('Feature Importance (Random Forest)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, max(importance_sorted) * 1.1]);

% Subplot 5: Correlation Matrix
subplot(2, 3, 5);
corr_data = [X_train, y_train];
corr_matrix = corr(corr_data);
feature_names_corr = [feature_names, {'ImpactStr'}];
imagesc(corr_matrix);
colorbar;
colormap(jet);
caxis([-1, 1]);
set(gca, 'XTick', 1:length(feature_names_corr));
set(gca, 'YTick', 1:length(feature_names_corr));
set(gca, 'XTickLabel', feature_names_corr, 'XTickLabelRotation', 45, 'FontSize', 9);
set(gca, 'YTickLabel', feature_names_corr, 'FontSize', 9);
title('Correlation Matrix', 'FontSize', 12, 'FontWeight', 'bold');

% Subplot 6: Error Comparison
subplot(2, 3, 6);
abs_errors = mean(abs(residuals), 1);
bar(1:5, abs_errors, 'FaceColor', [0.8, 0.2, 0.2]);
set(gca, 'XTick', 1:5);
set(gca, 'XTickLabel', model_names, 'FontSize', 10);
ylabel('Mean Absolute Error (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
title('Prediction Error Comparison', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
[~, best_idx] = min(abs_errors);
hold on;
bar(best_idx, abs_errors(best_idx), 'FaceColor', [0.2, 0.8, 0.2]);
hold off;

sgtitle('Figure 1: Impact Strength Prediction - Model Performance Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% 5. PARAMETER OPTIMIZATION FOR FIGURE 2
fprintf('Running parameter optimization for Figure 2...\n');

% Define parameter ranges
param_ranges.layer_thickness = [0.15, 0.2, 0.25];
param_ranges.infill_density = [20, 40, 60, 80, 100];
param_ranges.filling_pattern = [1, 2, 3];
param_ranges.nozzle_temp = [195, 200, 205, 210];
param_ranges.printing_speed = [30, 40, 50, 60];

% Store optimization results
optim_results = [];
counter = 0;

% Grid search
for lt = param_ranges.layer_thickness
    for id = param_ranges.infill_density
        for fp = param_ranges.filling_pattern
            for nt = param_ranges.nozzle_temp
                for ps = param_ranges.printing_speed
                    counter = counter + 1;
                    
                    % Create parameter vector
                    params = [lt, id, fp, 1.75, 0.4, nt, 60, ps*60, 1];
                    
                    % Predict using best model (Gradient Boosting)
                    params_norm = (params - mu) ./ sigma;
                    pred_norm = predict(gb_model, params_norm);
                    strength = pred_norm * sigma_y + mu_y;
                    
                    % Calculate print time (simplified)
                    layer_count = 10 / lt;  % 10mm height
                    perimeter_time = 200 / ps;  % seconds
                    infill_time = (id/100) * 300 / ps;
                    total_time_min = (perimeter_time + infill_time) / 60;
                    
                    % Store result
                    result = struct();
                    result.params = params;
                    result.strength = strength;
                    result.time_min = total_time_min;
                    result.layer = lt;
                    result.infill = id;
                    result.pattern = fp;
                    result.temp = nt;
                    result.speed = ps;
                    
                    optim_results = [optim_results; result];
                end
            end
        end
    end
end

fprintf('Optimization complete: %d configurations analyzed\n', length(optim_results));

%% 6. FIGURE 2: OPTIMIZATION ANALYSIS (6 subplots)
fprintf('\nGenerating Figure 2: Optimization Analysis...\n');

figure('Position', [50, 50, 1400, 800], 'Name', 'Parameter Optimization Analysis', 'NumberTitle', 'off');

% Extract data
strengths = [optim_results.strength];
times = [optim_results.time_min];
layers = [optim_results.layer];
infills = [optim_results.infill];
temps = [optim_results.temp];
speeds = [optim_results.speed];
patterns = [optim_results.pattern];

% Subplot 1: Pareto Front
subplot(2, 3, 1);
scatter(times, strengths, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Print Time (minutes)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Impact Strength (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
title('Strength vs Print Time Trade-off', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;
hold on;

% Find Pareto front
pareto_idx = [];
for i = 1:length(optim_results)
    dominated = false;
    for j = 1:length(optim_results)
        if i ~= j
            if (optim_results(j).strength >= optim_results(i).strength && ...
                optim_results(j).time_min <= optim_results(i).time_min) && ...
               (optim_results(j).strength > optim_results(i).strength || ...
                optim_results(j).time_min < optim_results(i).time_min)
                dominated = true;
                break;
            end
        end
    end
    if ~dominated
        pareto_idx = [pareto_idx, i];
    end
end

% Plot Pareto front
if ~isempty(pareto_idx)
    pareto_times = times(pareto_idx);
    pareto_strengths = strengths(pareto_idx);
    [pareto_times_sorted, sort_idx] = sort(pareto_times);
    pareto_strengths_sorted = pareto_strengths(sort_idx);
    plot(pareto_times_sorted, pareto_strengths_sorted, 'r-', 'LineWidth', 2);
    scatter(pareto_times, pareto_strengths, 80, 'r', 'filled');
    
    % Mark best points
    [max_strength, max_idx] = max(pareto_strengths);
    max_time = pareto_times(max_idx);
    scatter(max_time, max_strength, 120, 'g', '^', 'filled', 'LineWidth', 2);
    
    [min_time, min_idx] = min(pareto_times);
    min_strength = pareto_strengths(min_idx);
    scatter(min_time, min_strength, 120, 'm', 's', 'filled', 'LineWidth', 2);
    
    legend('All Configurations', 'Pareto Front', 'Pareto Optimal', ...
           'Max Strength', 'Fastest', 'Location', 'best', 'FontSize', 9);
else
    legend('All Configurations', 'Location', 'best', 'FontSize', 9);
end
hold off;

% Subplot 2: Strength vs Infill Density
subplot(2, 3, 2);
scatter(infills, strengths, 30, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Infill Density (%)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Impact Strength (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
title('Strength vs Infill Density', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% Add trend line
if length(unique(infills)) > 1
    p = polyfit(infills, strengths, 2);
    x_fit = linspace(min(infills), max(infills), 100);
    y_fit = polyval(p, x_fit);
    hold on;
    plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
    hold off;
    legend('Data', sprintf('Trend: y=%.3fx²+%.3fx+%.3f', p(1), p(2), p(3)), ...
           'Location', 'best', 'FontSize', 9);
else
    legend('Data', 'Location', 'best', 'FontSize', 9);
end

% Subplot 3: Strength vs Layer Thickness
subplot(2, 3, 3);
box_data = [];
layer_labels = {};
for i = 1:length(param_ranges.layer_thickness)
    lt_val = param_ranges.layer_thickness(i);
    idx = layers == lt_val;
    if any(idx)
        box_data = [box_data, strengths(idx)'];
        layer_labels{end+1} = sprintf('%.2fmm', lt_val);
    end
end
if ~isempty(box_data)
    boxplot(box_data, 'Labels', layer_labels);
    xlabel('Layer Thickness (mm)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Impact Strength (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Strength by Layer Thickness', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
end

% Subplot 4: Strength vs Nozzle Temperature
subplot(2, 3, 4);
scatter(temps, strengths, 30, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Nozzle Temperature (°C)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Impact Strength (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
title('Strength vs Nozzle Temperature', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% Subplot 5: Print Time vs Printing Speed
subplot(2, 3, 5);
scatter(speeds, times, 30, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('Printing Speed (mm/s)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Print Time (minutes)', 'FontSize', 11, 'FontWeight', 'bold');
title('Print Time vs Printing Speed', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% Add trend line
if length(unique(speeds)) > 1
    p = polyfit(speeds, times, 1);
    x_fit = linspace(min(speeds), max(speeds), 100);
    y_fit = polyval(p, x_fit);
    hold on;
    plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
    hold off;
    legend('Data', sprintf('Trend: y=%.3fx+%.3f', p(1), p(2)), ...
           'Location', 'best', 'FontSize', 9);
else
    legend('Data', 'Location', 'best', 'FontSize', 9);
end

% Subplot 6: Pattern Comparison
subplot(2, 3, 6);
pattern_names = {'Rectilinear', 'Triangular', 'Honeycomb'};
box_data = [];
for p = 1:3
    idx = patterns == p;
    if any(idx)
        box_data = [box_data, strengths(idx)'];
    end
end
if ~isempty(box_data)
    boxplot(box_data, 'Labels', pattern_names(1:size(box_data, 2)));
    xlabel('Infill Pattern', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Impact Strength (kJ/m²)', 'FontSize', 11, 'FontWeight', 'bold');
    title('Strength by Infill Pattern', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    set(gca, 'XTickLabelRotation', 45);
end

sgtitle('Figure 2: Parameter Optimization Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% 7. SUMMARY AND SAVE RESULTS
fprintf('\n=== SUMMARY OF RESULTS ===\n');
fprintf('Best Model: Gradient Boosting\n');
fprintf('  RMSE: %.4f kJ/m²\n', rmse_gb);
fprintf('  R²:   %.4f\n', r2_gb);
fprintf('\nOptimal Parameters from Optimization:\n');
if ~isempty(pareto_idx)
    [~, best_pareto_idx] = max(strengths(pareto_idx));
    best_config = optim_results(pareto_idx(best_pareto_idx));
    fprintf('  Layer: %.2f mm\n', best_config.layer);
    fprintf('  Infill: %d%%\n', best_config.infill);
    fprintf('  Pattern: %s\n', pattern_names{best_config.pattern});
    fprintf('  Nozzle Temp: %d°C\n', best_config.temp);
    fprintf('  Speed: %d mm/s\n', best_config.speed);
    fprintf('  Predicted Strength: %.2f kJ/m²\n', best_config.strength);
    fprintf('  Estimated Print Time: %.1f minutes\n', best_config.time_min);
end

% Save models and results
save('impact_strength_models_complete.mat', 'net_mlp', 'rf_model', 'gb_model', ...
     'tree_model', 'mu', 'sigma', 'mu_y', 'sigma_y', 'feature_names');

% Save optimization results
save('optimization_results.mat', 'optim_results', 'pareto_idx');

fprintf('\nAll figures generated successfully!\n');
fprintf('Models saved to: impact_strength_models_complete.mat\n');
fprintf('Optimization results saved to: optimization_results.mat\n');
fprintf('\n=== ANALYSIS COMPLETE ===\n');

%% 8. CREATE PREDICTION FUNCTION
prediction_code = {
    'function [prediction, all_predictions] = predictImpact(params, model_type)';
    '% Predict impact strength for 3D printing parameters';
    '% ';
    '% Input: params = [layer, infill, pattern, filament_d, nozzle_d, nozzle_t, bed_t, speed, color]';
    '%        model_type = ''mlp'', ''rf'', ''gb'', ''tree'', or ''ensemble''';
    '% ';
    '% Example:';
    '%   params = [0.25, 60, 1, 1.75, 0.4, 195, 60, 1800, 1];';
    '%   strength = predictImpact(params, ''gb'');';
    '% ';
    '    % Load models';
    '    load(''impact_strength_models_complete.mat'');';
    '    ';
    '    % Normalize input';
    '    params_norm = (params - mu) ./ sigma;';
    '    ';
    '    % Make predictions';
    '    switch lower(model_type)';
    '        case ''mlp''';
    '            pred_norm = net_mlp(params_norm'');';
    '        case ''rf''';
    '            pred_norm = predict(rf_model, params_norm);';
    '        case ''gb''';
    '            pred_norm = predict(gb_model, params_norm);';
    '        case ''tree''';
    '            pred_norm = predict(tree_model, params_norm);';
    '        case ''ensemble''';
    '            % Weighted average of all models';
    '            pred_mlp = net_mlp(params_norm'');';
    '            pred_rf = predict(rf_model, params_norm);';
    '            pred_gb = predict(gb_model, params_norm);';
    '            pred_tree = predict(tree_model, params_norm);';
    '            all_preds = [pred_mlp'', pred_rf, pred_gb, pred_tree];';
    '            weights = [0.1, 0.3, 0.4, 0.2];  % Based on performance';
    '            pred_norm = sum(all_preds .* weights, 2);';
    '            all_predictions = struct(''mlp'', pred_mlp''*sigma_y+mu_y, ...';
    '                                     ''rf'', pred_rf*sigma_y+mu_y, ...';
    '                                     ''gb'', pred_gb*sigma_y+mu_y, ...';
    '                                     ''tree'', pred_tree*sigma_y+mu_y);';
    '        otherwise';
    '            error(''Unknown model type'');';
    '    end';
    '    ';
    '    % Denormalize and return';
    '    prediction = pred_norm * sigma_y + mu_y;';
    'end'
};

% Save prediction function
fid = fopen('predictImpact.m', 'w');
for i = 1:length(prediction_code)
    fprintf(fid, '%s\n', prediction_code{i});
end
fclose(fid);

fprintf('Prediction function created: predictImpact.m\n');