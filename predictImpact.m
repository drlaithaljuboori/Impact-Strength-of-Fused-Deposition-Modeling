function [prediction, all_predictions] = predictImpact(params, model_type)
% Predict impact strength for 3D printing parameters
% 
% Input: params = [layer, infill, pattern, filament_d, nozzle_d, nozzle_t, bed_t, speed, color]
%        model_type = 'mlp', 'rf', 'gb', 'tree', or 'ensemble'
% 
% Example:
%   params = [0.25, 60, 1, 1.75, 0.4, 195, 60, 1800, 1];
%   strength = predictImpact(params, 'gb');
% 
    % Load models
    load('impact_strength_models_complete.mat');
    
    % Normalize input
    params_norm = (params - mu) ./ sigma;
    
    % Make predictions
    switch lower(model_type)
        case 'mlp'
            pred_norm = net_mlp(params_norm');
        case 'rf'
            pred_norm = predict(rf_model, params_norm);
        case 'gb'
            pred_norm = predict(gb_model, params_norm);
        case 'tree'
            pred_norm = predict(tree_model, params_norm);
        case 'ensemble'
            % Weighted average of all models
            pred_mlp = net_mlp(params_norm');
            pred_rf = predict(rf_model, params_norm);
            pred_gb = predict(gb_model, params_norm);
            pred_tree = predict(tree_model, params_norm);
            all_preds = [pred_mlp', pred_rf, pred_gb, pred_tree];
            weights = [0.1, 0.3, 0.4, 0.2];  % Based on performance
            pred_norm = sum(all_preds .* weights, 2);
            all_predictions = struct('mlp', pred_mlp'*sigma_y+mu_y, ...
                                     'rf', pred_rf*sigma_y+mu_y, ...
                                     'gb', pred_gb*sigma_y+mu_y, ...
                                     'tree', pred_tree*sigma_y+mu_y);
        otherwise
            error('Unknown model type');
    end
    
    % Denormalize and return
    prediction = pred_norm * sigma_y + mu_y;
end
