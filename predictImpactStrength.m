function [prediction, all_predictions, model_info] = predictImpactStrength(params, model_type)
% Predict impact strength from 3D printing parameters
% 
% Inputs:
%   params: [1x9] vector containing:
%           [LayerThickness, InfillDensity, FillingPattern,
%            FilamentDiameter, NozzleDiameter, NozzleTemp,
%            BedTemp, PrintingSpeed, FilamentColor]
%   model_type: 'mlp', 'dl', 'rf', 'svm', 'gb', 'tree', 'ensemble', or 'all'
% 
% Outputs:
%   prediction: Predicted impact strength (kJ/m²)
%   all_predictions: Structure with predictions from all models
%   model_info: Structure with model details and confidence intervals
% 
% Example:
%   params = [0.2, 50, 1, 1.75, 0.4, 200, 60, 3000, 1];
%   [strength, all_preds] = predictImpactStrength(params, 'ensemble');
% 

    % Load saved models
    if ~exist('impact_strength_models.mat', 'file')
        error('Model file not found. Run the training script first.');
    end
    
    load('impact_strength_models.mat', 'models_collection');
    
    % Check input dimensions
    if length(params) ~= 9
        error('Input must have exactly 9 parameters');
    end
    
    % Normalize input
    params_norm = (params - models_collection.mu) ./ models_collection.sigma;
    
    % Initialize outputs
    all_predictions = struct();
    model_info = struct();
    
    % Make predictions with all models
    
    % 1. MLP
    try
        pred_norm_mlp = models_collection.mlp(params_norm');
        all_predictions.mlp = pred_norm_mlp' * models_collection.sigma_y + models_collection.mu_y;
    catch
        all_predictions.mlp = NaN;
    end
    
    % 2. Deep Learning
    try
        if isa(models_collection.dl, 'RegressionNeuralNetwork')
            pred_norm_dl = predict(models_collection.dl, params_norm);
        else
            pred_norm_dl = models_collection.dl(params_norm');
            if size(pred_norm_dl, 1) > 1
                pred_norm_dl = pred_norm_dl';
            end
        end
        all_predictions.dl = pred_norm_dl * models_collection.sigma_y + models_collection.mu_y;
    catch
        all_predictions.dl = NaN;
    end
    
    % 3. Random Forest
    try
        pred_norm_rf = predict(models_collection.random_forest, params_norm);
        all_predictions.random_forest = pred_norm_rf * models_collection.sigma_y + models_collection.mu_y;
    catch
        all_predictions.random_forest = NaN;
    end
    
    % 4. SVM
    try
        pred_norm_svm = predict(models_collection.svm, params_norm);
        all_predictions.svm = pred_norm_svm * models_collection.sigma_y + models_collection.mu_y;
    catch
        all_predictions.svm = NaN;
    end
    
    % 5. Gradient Boosting
    try
        pred_norm_gb = predict(models_collection.gradient_boosting, params_norm);
        all_predictions.gradient_boosting = pred_norm_gb * models_collection.sigma_y + models_collection.mu_y;
    catch
        all_predictions.gradient_boosting = NaN;
    end
    
    % 6. Decision Tree
    try
        pred_norm_tree = predict(models_collection.decision_tree, params_norm);
        all_predictions.decision_tree = pred_norm_tree * models_collection.sigma_y + models_collection.mu_y;
    catch
        all_predictions.decision_tree = NaN;
    end
    
    % Calculate ensemble prediction (weighted average)
    valid_preds = [all_predictions.mlp, all_predictions.dl, all_predictions.random_forest, ...
                   all_predictions.svm, all_predictions.gradient_boosting, all_predictions.decision_tree];
    valid_mask = ~isnan(valid_preds);
    
    if any(valid_mask)
        valid_weights = models_collection.ensemble_weights(valid_mask);
        valid_weights = valid_weights / sum(valid_weights);
        ensemble_pred = sum(valid_preds(valid_mask) .* valid_weights);
    else
        ensemble_pred = NaN;
    end
    
    all_predictions.ensemble = ensemble_pred;
    
    % Return prediction based on requested model type
    switch lower(model_type)
        case 'mlp'
            prediction = all_predictions.mlp;
        case 'dl'
            prediction = all_predictions.dl;
        case 'rf'
            prediction = all_predictions.random_forest;
        case 'svm'
            prediction = all_predictions.svm;
        case 'gb'
            prediction = all_predictions.gradient_boosting;
        case 'tree'
            prediction = all_predictions.decision_tree;
        case 'ensemble'
            prediction = all_predictions.ensemble;
        case 'all'
            % Return all predictions
            prediction = all_predictions;
        otherwise
            error('Unknown model type. Use: mlp, dl, rf, svm, gb, tree, ensemble, or all');
    end
    
    % Calculate confidence intervals (simple method)
    if nargout > 2
        model_info.timestamp = datetime('now');
        model_info.input_parameters = params;
        model_info.normalized_input = params_norm;
        
        % Calculate prediction range
        all_pred_values = [all_predictions.mlp, all_predictions.dl, ...
                           all_predictions.random_forest, all_predictions.svm, ...
                           all_predictions.gradient_boosting, all_predictions.decision_tree];
        model_info.prediction_range = [min(all_pred_values), max(all_pred_values)];
        model_info.prediction_std = std(all_pred_values(~isnan(all_pred_values)));
        
        % Calculate confidence interval (95%)
        if ~isnan(ensemble_pred)
            model_info.confidence_interval = [ensemble_pred - 1.96*model_info.prediction_std, ...
                                              ensemble_pred + 1.96*model_info.prediction_std];
        else
            model_info.confidence_interval = [NaN, NaN];
        end
    end
end
