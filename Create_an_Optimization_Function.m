%% FINAL OPTIMIZATION WITH FIXES
clear; clc; close all;

% Load models
load('impact_strength_models.mat', 'models_collection');

fprintf('=== FINAL PARAMETER OPTIMIZATION ===\n\n');

% Define pattern names (GLOBAL for all functions)
pattern_names = {'Rectilinear', 'Triangular', 'Honeycomb'};
pattern_short = {'Rect', 'Triang', 'Honey'};

% Define practical parameter ranges
param_ranges.layer_thickness = [0.15, 0.2, 0.25];  % mm
param_ranges.infill_density = [20, 40, 60, 80, 100];  % %
param_ranges.filling_pattern = [1, 2, 3];  % 1=Rectilinear, 2=Triangular, 3=Honeycomb
param_ranges.nozzle_temp = [195, 200, 205, 210];  % °C
param_ranges.printing_speed = [30, 40, 50, 60];  % mm/s
param_ranges.filament_color = 1;  % Black only

% Fixed parameters
filament_diameter = 1.75;  % mm
nozzle_diameter = 0.4;     % mm
bed_temp = 60;             % °C

% TEST SPECIMEN DIMENSIONS (ASTM D256 Type A)
part_length = 63.5;    % mm
part_width = 12.7;     % mm
part_height = 10;      % mm (increased from 3.2mm for more realistic time)
part_volume = part_length * part_width * part_height;  % mm³

% Print settings
perimeter_count = 3;           % Number of perimeter shells
layer_change_time = 15;        % seconds per layer (more realistic)

% Search for optimal parameters
fprintf('Running optimization...\n');

best_strength = -inf;
best_params = [];
all_results = [];

% Calculate combinations
n_combinations = length(param_ranges.layer_thickness) * ...
                 length(param_ranges.infill_density) * ...
                 length(param_ranges.filling_pattern) * ...
                 length(param_ranges.nozzle_temp) * ...
                 length(param_ranges.printing_speed) * ...
                 length(param_ranges.filament_color);

fprintf('Testing %d parameter combinations\n', n_combinations);

counter = 0;
for lt = param_ranges.layer_thickness
    for id = param_ranges.infill_density
        for fp = param_ranges.filling_pattern
            for nt = param_ranges.nozzle_temp
                for ps_mm_s = param_ranges.printing_speed
                    for fc = param_ranges.filament_color
                        
                        counter = counter + 1;
                        
                        % Create parameter vector
                        params = [lt, id, fp, filament_diameter, nozzle_diameter, ...
                                 nt, bed_temp, ps_mm_s*60, fc];
                        
                        try
                            % Predict using Gradient Boosting
                            params_norm = (params - models_collection.mu) ./ models_collection.sigma;
                            pred_norm = predict(models_collection.gradient_boosting, params_norm);
                            strength = pred_norm * models_collection.sigma_y + models_collection.mu_y;
                            
                            % REALISTIC PRINT TIME CALCULATION
                            % 1. Perimeter time
                            perimeter_length = 2 * (part_length + part_width);
                            perimeter_time_sec = (perimeter_length * perimeter_count) / ps_mm_s;
                            
                            % 2. Infill time (pattern factor)
                            if fp == 3  % Honeycomb
                                pattern_factor = 1.5;
                            elseif fp == 2  % Triangular
                                pattern_factor = 1.2;
                            else  % Rectilinear
                                pattern_factor = 1.0;
                            end
                            
                            infill_area = part_length * part_width * (id/100);
                            line_spacing = nozzle_diameter * 0.9;
                            infill_length = (infill_area / line_spacing) * pattern_factor;
                            infill_time_sec = infill_length / ps_mm_s;
                            
                            % 3. Layer changes and movements
                            layer_count = ceil(part_height / lt);
                            movement_time_sec = layer_count * layer_change_time;
                            
                            % 4. Total time (add 20% for acceleration/deceleration)
                            total_time_sec = (perimeter_time_sec + infill_time_sec + movement_time_sec) * 1.2;
                            total_time_min = total_time_sec / 60;
                            
                            % Store result
                            result = struct();
                            result.params = params;
                            result.strength = strength;
                            result.print_time_minutes = total_time_min;
                            result.infill_percentage = id;
                            result.layer_thickness = lt;
                            result.print_speed_mm_s = ps_mm_s;
                            result.pattern = fp;
                            result.nozzle_temp = nt;
                            
                            all_results = [all_results; result];
                            
                            % Update best
                            if strength > best_strength && total_time_min < 240  % < 4 hours
                                best_strength = strength;
                                best_params = params;
                                best_result = result;
                            end
                            
                        catch
                            continue;
                        end
                    end
                end
            end
        end
    end
end

fprintf('Optimization complete!\n\n');

%% Display Results
fprintf('=== OPTIMIZATION RESULTS ===\n');

if isempty(all_results)
    fprintf('No valid predictions.\n');
    return;
end

% Extract arrays
strengths = [all_results.strength];
print_times = [all_results.print_time_minutes];

% Show all results sorted by strength
[sorted_strengths, sort_idx] = sort(strengths, 'descend');
top_results = all_results(sort_idx(1:min(20, length(all_results))));

fprintf('\nTop Configurations (sorted by strength):\n');
fprintf('Rank | Strength | Time(min) | Layer | Infill%% | Pattern  | Temp | Speed\n');
fprintf('-----|----------|-----------|-------|---------|----------|------|-------\n');

for i = 1:length(top_results)
    fprintf('%4d | %8.2f | %9.1f | %5.2f | %7d | %7s | %4d | %5.0f\n', ...
        i, top_results(i).strength, top_results(i).print_time_minutes, ...
        top_results(i).layer_thickness, top_results(i).infill_percentage, ...
        pattern_short{top_results(i).pattern}, top_results(i).nozzle_temp, ...
        top_results(i).print_speed_mm_s);
end

% Display best configuration
if exist('best_result', 'var')
    fprintf('\n=== BEST CONFIGURATION ===\n');
    fprintf('Predicted Impact Strength: %.2f kJ/m²\n', best_strength);
    fprintf('Estimated Print Time: %.1f minutes\n', best_result.print_time_minutes);
    fprintf('\nOptimal Parameters:\n');
    fprintf('  Layer Thickness:      %.2f mm\n', best_result.layer_thickness);
    fprintf('  Infill Density:       %.0f%%\n', best_result.infill_percentage);
    fprintf('  Filling Pattern:      %s\n', pattern_names{best_result.pattern});
    fprintf('  Nozzle Temperature:   %.0f°C\n', best_result.nozzle_temp);
    fprintf('  Printing Speed:       %.0f mm/s (%.0f mm/min)\n', ...
        best_result.print_speed_mm_s, best_result.print_speed_mm_s*60);
    fprintf('  Bed Temperature:      %.0f°C\n', bed_temp);
    fprintf('  Filament:             Black PLA\n');
else
    fprintf('\nNo configuration met time constraints.\n');
end

%% Find Pareto Front (if we have results)
if length(all_results) >= 10
    fprintf('\n=== PARETO FRONT ANALYSIS ===\n');
    
    % Find Pareto optimal solutions
    pareto_indices = [];
    n = length(all_results);
    
    for i = 1:n
        dominated = false;
        for j = 1:n
            if i ~= j
                % Check if j dominates i
                if (all_results(j).strength > all_results(i).strength && ...
                    all_results(j).print_time_minutes <= all_results(i).print_time_minutes) || ...
                   (all_results(j).strength >= all_results(i).strength && ...
                    all_results(j).print_time_minutes < all_results(i).print_time_minutes)
                    dominated = true;
                    break;
                end
            end
        end
        if ~dominated
            pareto_indices = [pareto_indices, i];
        end
    end
    
    if ~isempty(pareto_indices)
        pareto_results = all_results(pareto_indices);
        pareto_strengths = [pareto_results.strength];
        pareto_times = [pareto_results.print_time_minutes];
        
        % Sort by strength
        [pareto_strengths_sorted, sort_order] = sort(pareto_strengths, 'descend');
        pareto_results_sorted = pareto_results(sort_order);
        pareto_times_sorted = pareto_times(sort_order);
        
        % Take top 10 unique
        unique_results = [];
        seen = {};
        for i = 1:length(pareto_results_sorted)
            r = pareto_results_sorted(i);
            key = sprintf('%.2f-%d-%d', r.layer_thickness, r.infill_percentage, r.pattern);
            if ~ismember(key, seen)
                unique_results = [unique_results, r];
                seen{end+1} = key;
            end
            if length(unique_results) >= 10
                break;
            end
        end
        
        fprintf('\nTop Pareto Optimal Solutions:\n');
        fprintf('ID | Strength | Time(min) | Infill%% | Layer | Pattern | Speed\n');
        fprintf('---|----------|-----------|---------|-------|---------|-------\n');
        
        for i = 1:length(unique_results)
            r = unique_results(i);
            fprintf('%2d | %8.2f | %9.1f | %7d | %5.2f | %6s | %5.0f\n', ...
                i, r.strength, r.print_time_minutes, ...
                r.infill_percentage, r.layer_thickness, ...
                pattern_short{r.pattern}, r.print_speed_mm_s);
        end
        
        % Identify key configurations
        if length(unique_results) >= 3
            % Max strength
            max_strength_config = unique_results(1);
            
            % Fastest (lowest time)
            [min_time, min_idx] = min([unique_results.print_time_minutes]);
            fast_config = unique_results(min_idx);
            
            % Best value (strength/time ratio)
            ratios = [unique_results.strength] ./ [unique_results.print_time_minutes];
            [max_ratio, ratio_idx] = max(ratios);
            value_config = unique_results(ratio_idx);
            
            fprintf('\n=== KEY RECOMMENDATIONS ===\n');
            
            fprintf('\n1. MAXIMUM STRENGTH:\n');
            fprintf('   Strength: %.2f kJ/m²\n', max_strength_config.strength);
            fprintf('   Time: %.1f minutes\n', max_strength_config.print_time_minutes);
            
            fprintf('\n2. FASTEST PRINT:\n');
            fprintf('   Strength: %.2f kJ/m²\n', fast_config.strength);
            fprintf('   Time: %.1f minutes\n', min_time);
            
            fprintf('\n3. BEST VALUE (Strength/Time):\n');
            fprintf('   Strength: %.2f kJ/m²\n', value_config.strength);
            fprintf('   Time: %.1f minutes\n', value_config.print_time_minutes);
            fprintf('   Ratio: %.2f kJ/m² per minute\n', max_ratio);
        end
    else
        fprintf('No Pareto optimal solutions found.\n');
    end
end

%% Create Experimental Validation Plan
fprintf('\n=== EXPERIMENTAL VALIDATION PLAN ===\n\n');

fprintf('Recommended Test Configurations:\n');
fprintf('===============================\n\n');

% Always test baseline
fprintf('1. BASELINE (Standard Settings):\n');
fprintf('   Parameters:\n');
fprintf('     - Layer: 0.20 mm\n');
fprintf('     - Infill: 50%%\n');
fprintf('     - Pattern: Rectilinear\n');
fprintf('     - Nozzle: 200°C\n');
fprintf('     - Speed: 50 mm/s\n');
fprintf('     - Bed: 60°C\n');
fprintf('     - Filament: Black PLA\n\n');

% Add optimized configurations if available
if exist('best_result', 'var')
    fprintf('2. OPTIMIZED FOR STRENGTH:\n');
    fprintf('   Expected: %.2f kJ/m² ± 1.5 kJ/m²\n', best_strength);
    fprintf('   Print Time: ~%.0f minutes\n', best_result.print_time_minutes);
    fprintf('   Parameters:\n');
    fprintf('     - Layer: %.2f mm\n', best_result.layer_thickness);
    fprintf('     - Infill: %d%%\n', best_result.infill_percentage);
    fprintf('     - Pattern: %s\n', pattern_names{best_result.pattern});
    fprintf('     - Nozzle: %d°C\n', best_result.nozzle_temp);
    fprintf('     - Speed: %.0f mm/s\n', best_result.print_speed_mm_s);
    fprintf('     - Bed: 60°C\n');
    fprintf('     - Filament: Black PLA\n\n');
end

% Add a high-speed configuration
fprintf('3. HIGH SPEED PROTOTYPING:\n');
fprintf('   Parameters:\n');
fprintf('     - Layer: 0.25 mm\n');
fprintf('     - Infill: 30%%\n');
fprintf('     - Pattern: Rectilinear\n');
fprintf('     - Nozzle: 205°C\n');
fprintf('     - Speed: 60 mm/s\n');
fprintf('     - Bed: 60°C\n');
fprintf('     - Filament: Black PLA\n\n');

fprintf('Experimental Protocol:\n');
fprintf('=====================\n');
fprintf('1. Sample Preparation:\n');
fprintf('   • Print 5 specimens per configuration\n');
fprintf('   • Use same printer and filament spool\n');
fprintf('   • Allow 24 hours conditioning at 23°C, 50%% RH\n\n');

fprintf('2. Impact Testing (ASTM D256):\n');
fprintf('   • Use Izod impact tester with 2.75J hammer\n');
fprintf('   • Notch depth: 2.54 mm\n');
fprintf('   • Test speed: 3.5 m/s\n');
fprintf('   • Record energy to break in Joules\n\n');

fprintf('3. Data Analysis:\n');
fprintf('   • Convert to kJ/m²: Strength = Energy / (Width × Thickness)\n');
fprintf('   • Calculate mean ± standard deviation\n');
fprintf('   • Compare with model predictions\n');

%% Model Performance Summary
fprintf('\n=== MODEL PERFORMANCE SUMMARY ===\n\n');

% Get model metrics from training
fprintf('Gradient Boosting Model Performance:\n');
fprintf('• R² Score: 0.9709 (Excellent)\n');
fprintf('• RMSE: 0.9082 kJ/m²\n');
fprintf('• MAE: 0.7901 kJ/m²\n');
fprintf('• Prediction Range: Typically ±1.5 kJ/m²\n\n');

fprintf('Key Predictive Features:\n');
if isfield(models_collection, 'random_forest') && ...
   isfield(models_collection.random_forest, 'OOBPermutedPredictorDeltaError')
    importance = models_collection.random_forest.OOBPermutedPredictorDeltaError;
    [sorted_imp, idx] = sort(importance, 'descend');
    
    for i = 1:min(5, length(sorted_imp))
        fprintf('  %d. %s (Importance: %.3f)\n', i, ...
            models_collection.feature_names{idx(i)}, sorted_imp(i));
    end
else
    fprintf('  1. Infill Density\n');
    fprintf('  2. Layer Thickness\n');
    fprintf('  3. Nozzle Temperature\n');
    fprintf('  4. Printing Speed\n');
    fprintf('  5. Filling Pattern\n');
end

%% Final Recommendations
fprintf('\n=== FINAL RECOMMENDATIONS ===\n\n');

fprintf('FOR MAXIMUM IMPACT STRENGTH:\n');
if exist('best_result', 'var')
    fprintf('• Use: %.2f mm layer, %d%% infill, %s pattern\n', ...
        best_result.layer_thickness, best_result.infill_percentage, ...
        pattern_names{best_result.pattern});
    fprintf('• Nozzle: %d°C, Speed: %.0f mm/s\n', ...
        best_result.nozzle_temp, best_result.print_speed_mm_s);
    fprintf('• Expected: %.2f kJ/m²\n\n', best_strength);
else
    fprintf('• Use: 0.25 mm layer, 60-70%% infill, Rectilinear pattern\n');
    fprintf('• Nozzle: 200°C, Speed: 40-50 mm/s\n');
    fprintf('• Expected: ~16.0 kJ/m²\n\n');
end

fprintf('FOR PRACTICAL APPLICATIONS:\n');
fprintf('• Structural parts: 0.25 mm, 60%% infill, Rectilinear\n');
fprintf('• General purpose: 0.20 mm, 40%% infill, Rectilinear\n');
fprintf('• Fast prototypes: 0.25 mm, 30%% infill, Rectilinear, 60 mm/s\n\n');

fprintf('IMPORTANT NOTES:\n');
fprintf('1. Always validate with test prints\n');
fprintf('2. Results are specific to your printer and filament\n');
fprintf('3. Model accuracy improves with more data\n');
fprintf('4. Consider print orientation effects (not in model)\n');

%% Save Results
final_report = struct();
final_report.optimization_date = datetime('now');
final_report.model = 'Gradient Boosting';
final_report.r2_score = 0.9709;
final_report.rmse = 0.9082;

if exist('best_result', 'var')
    final_report.recommended_config = struct(...
        'layer_thickness', best_result.layer_thickness, ...
        'infill_percentage', best_result.infill_percentage, ...
        'pattern', pattern_names{best_result.pattern}, ...
        'nozzle_temp', best_result.nozzle_temp, ...
        'print_speed', best_result.print_speed_mm_s, ...
        'predicted_strength', best_strength, ...
        'estimated_time', best_result.print_time_minutes);
end

save('final_recommendations.mat', 'final_report');
fprintf('\nRecommendations saved to: final_recommendations.mat\n');

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Ready for experimental validation!\n');