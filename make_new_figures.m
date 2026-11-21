%% make_new_figures.m
% Generates corrected/new figures for the revised manuscript:
%   Figure 7  (replacement: separate RMSE / R^2 axes)
%   Figure 8  (corrected: one-hot encoded feature importance)
%   Figure 11 (corrected: total print time in minutes)
%   Figure 12 (new: Pareto front, restricted domain)
%   Figure 13 (new: Pareto sensitivity to time weighting)
clear; clc; close all;

%% ---------- Load models ----------
if exist('impact_strength_models.mat','file')
    S = load('impact_strength_models.mat'); MC = S.models_collection;
    gb = MC.gradient_boosting; rf = MC.random_forest;
    net_mlp = MC.mlp; tree_model = MC.decision_tree;
    mu = MC.mu; sigma = MC.sigma; mu_y = MC.mu_y; sigma_y = MC.sigma_y;
elseif exist('impact_strength_models_complete.mat','file')
    S = load('impact_strength_models_complete.mat');
    gb = S.gb_model; rf = S.rf_model; net_mlp = S.net_mlp; tree_model = S.tree_model;
    mu = S.mu; sigma = S.sigma; mu_y = S.mu_y; sigma_y = S.sigma_y;
else
    error('No model file found.');
end

%% ---------- Rebuild dataset ----------
data = [
 0.15 30 10.9 1 1.75 0.4 200 60 3000 1; 0.15 50 12.36 1 1.75 0.4 200 60 3000 1;
 0.15 70 13.7 1 1.75 0.4 200 60 3000 1; 0.2 30 12.96 1 1.75 0.4 200 60 3000 1;
 0.2 50 13.73 1 1.75 0.4 200 60 3000 1; 0.2 70 16.13 1 1.75 0.4 200 60 3000 1;
 0.25 30 14.02 1 1.75 0.4 200 60 3000 1; 0.25 50 15 1 1.75 0.4 200 60 3000 1;
 0.25 70 17.04 1 1.75 0.4 200 60 3000 1; 0.15 30 10.5 1 1.75 0.4 200 60 3000 1;
 0.15 50 11.26 1 1.75 0.4 200 60 3000 1; 0.15 70 12.13 1 1.75 0.4 200 60 3000 1;
 0.2 30 11.38 1 1.75 0.4 200 60 3000 1; 0.2 50 12.46 1 1.75 0.4 200 60 3000 1;
 0.2 70 13.55 1 1.75 0.4 200 60 3000 1; 0.25 30 13.86 1 1.75 0.4 200 60 3000 1;
 0.25 50 14.96 1 1.75 0.4 200 60 3000 1; 0.25 70 16.4 1 1.75 0.4 200 60 3000 1;
 0.2 100 4.2 1 1.75 0.4 220 60 2400 2; 0.2 75 2.9 1 1.75 0.4 220 60 2400 2;
 0.2 50 2.42 1 1.75 0.4 220 60 2400 3; 0.2 75 1.7 1 1.75 0.4 220 60 2400 3;
 0.2 100 1.85 1 1.75 0.4 220 60 2400 4; 0.2 50 2.58 1 1.75 0.4 220 60 2400 4;
 0.2 100 3.15 1 1.75 0.4 220 60 2400 5; 0.2 50 2.75 1 1.75 0.4 220 60 2400 6;
 0.2 75 3.54 1 1.75 0.4 220 60 2400 7; 0.2 10 1.79 1 1.75 0.4 205 60 3600 1;
 0.2 30 2.291 1 1.75 0.4 205 60 3600 1; 0.2 50 1.921 1 1.75 0.4 205 60 3600 1;
 0.2 10 1.79 2 1.75 0.4 205 60 3600 1; 0.2 30 1.799 2 1.75 0.4 205 60 3600 1;
 0.2 50 3.571 2 1.75 0.4 205 60 3600 1; 0.2 10 1.432 3 1.75 0.4 205 60 3600 1;
 0.2 30 2.694 3 1.75 0.4 205 60 3600 1; 0.2 50 3.249 3 1.75 0.4 205 60 3600 1];
X = data(:,[1,2,4:10]); y = data(:,3);
rng(42); cv = cvpartition(length(y),'HoldOut',0.2);
Xtr = X(training(cv),:); ytr = y(training(cv),:);
Xte = X(test(cv),:);    yte = y(test(cv),:);

%% ================= FIGURE 8 (corrected feature importance) =================
Xtr_oh = onehot_encode(Xtr); feat_oh = onehot_feat_names();
rf_oh = TreeBagger(100, Xtr_oh, ytr, 'Method','regression', ...
    'OOBPredictorImportance','on','MinLeafSize',5);
imp = rf_oh.OOBPermutedPredictorDeltaError;
[imp_s, idx] = sort(imp,'descend');

figure('Position',[80 80 900 520],'Name','Figure 8 corrected');
barh(imp_s,'FaceColor',[0.2 0.4 0.8]);
set(gca,'YTick',1:numel(imp_s),'YTickLabel',feat_oh(idx),'FontSize',10);
xlabel('OOB Permuted Predictor Importance','FontWeight','bold');
title('Figure 8: Feature Importance (one-hot encoded)','FontWeight','bold');
grid on; xlim([0 max(imp_s)*1.1]);
exportgraphics(gcf,'Figure8_corrected.png','Resolution',300);

%% ================= FIGURE 11 + 12 (restricted-domain grid) =================
grid_lt = [0.15 0.20 0.25]; grid_id = [20 40 60 80 100];
grid_fp = [1 2 3]; grid_nt = [200 205 220]; grid_ps = [40 50 60]; % mm/s
T = []; Str = [];
for lt=grid_lt, for id=grid_id, for fp=grid_fp, for nt=grid_nt, for ps=grid_ps
    p = [lt id fp 1.75 0.4 nt 60 ps*60 1];
    pn = (p-mu)./sigma; sn = predict(gb,pn); s = sn*sigma_y+mu_y;
    T(end+1,1) = print_time_min(lt,id,fp,ps);
    Str(end+1,1) = s;
end,end,end,end,end

figure('Position',[80 80 900 520],'Name','Figure 11 corrected');
scatter(T,Str,25,'filled','MarkerFaceAlpha',0.35); grid on; box on;
xlabel('Total print time (min)','FontWeight','bold');
ylabel('Predicted impact strength (kJ/m^2)','FontWeight','bold');
title('Figure 11: Print time vs. impact strength (restricted domain)','FontWeight','bold');
exportgraphics(gcf,'Figure11_corrected.png','Resolution',300);

dom = false(size(T));
for i=1:numel(T)
    for j=1:numel(T)
        if i~=j && ((Str(j)>=Str(i)&&T(j)<=T(i)) && ...
                    (Str(j)>Str(i)||T(j)<T(i)))
            dom(i)=true; break;
        end
    end
end
pIdx = find(~dom); [Ts,ord] = sort(T(pIdx)); Ss = Str(pIdx(ord));

figure('Position',[80 80 900 520],'Name','Figure 12 Pareto');
scatter(T,Str,20,'filled','MarkerFaceAlpha',0.25); hold on;
plot(Ts,Ss,'r-','LineWidth',2); scatter(Ts,Ss,70,'r','filled');
[~,mi]=max(Ss); scatter(Ts(mi),Ss(mi),140,'g','^','filled','LineWidth',2);
[~,ti]=min(Ts); scatter(Ts(ti),Ss(ti),140,'m','s','filled','LineWidth',2);
grid on; box on;
legend({'All','Pareto','Pareto pts','Max strength','Fastest'},'Location','best');
xlabel('Total print time (min)','FontWeight','bold');
ylabel('Predicted impact strength (kJ/m^2)','FontWeight','bold');
title('Figure 12: Pareto front (restricted domain)','FontWeight','bold');
exportgraphics(gcf,'Figure12_pareto.png','Resolution',300);

%% ================= FIGURE 13 (sensitivity to time weighting) =================
w = [0.25 0.5 1 2 4];
figure('Position',[80 80 1200 420],'Name','Figure 13 sensitivity');
for k=1:numel(w)
    subplot(1,numel(w),k);
    score = Str - w(k)*0.05*T;
    [~,bi]=max(score);
    scatter(T,Str,15,'filled','MarkerFaceAlpha',0.2); hold on;
    scatter(T(bi),Str(bi),120,'r','filled');
    grid on; box on; title(sprintf('w_t = %.2f',w(k)));
    xlabel('Time (min)'); ylabel('Strength (kJ/m^2)');
end
sgtitle('Figure 13: Pareto sensitivity to print-time weighting','FontWeight','bold');
exportgraphics(gcf,'Figure13_sensitivity.png','Resolution',300);

%% ================= FIGURE 7 (replacement: separate axes) =================
[~,muX]=zscore(Xtr); sigX=std(Xtr); XteN=(Xte-muX)./sigX;
[~,muY]=zscore(ytr); sigY=std(ytr);
preds = {net_mlp(XteN')'*sigY+muY, predict(rf,XteN)*sigY+muY, ...
         predict(gb,XteN)*sigY+muY, predict(tree_model,XteN)*sigY+muY};
models = {'MLP','RF','GB','Tree'}; RMSE = zeros(1,4); R2 = zeros(1,4);
for i=1:4
    e = yte - preds{i}; RMSE(i)=sqrt(mean(e.^2));
    R2(i)=1-sum(e.^2)/sum((yte-mean(yte)).^2);
end
figure('Position',[80 80 1000 420],'Name','Figure 7 replacement');
subplot(1,2,1); bar(RMSE,'FaceColor',[0.8 0.3 0.3]);
set(gca,'XTickLabel',models); ylabel('RMSE (kJ/m^2)'); grid on;
title('RMSE (lower is better)','FontWeight','bold');
subplot(1,2,2); bar(R2,'FaceColor',[0.2 0.6 0.3]);
set(gca,'XTickLabel',models); ylabel('R^2'); ylim([0 1.1]); grid on;
title('R^2 (higher is better)','FontWeight','bold');
sgtitle('Figure 7: Model comparison (separate axes)','FontWeight','bold');
exportgraphics(gcf,'Figure7_replacement.png','Resolution',300);

%% ---------- Local functions ----------
function T = print_time_min(lt,id,fp,ps)
    L=63.5; W=12.7; H=10; Np=3; tl=15; nd=0.4;
    if fp==3, kp=1.5; elseif fp==2, kp=1.2; else, kp=1.0; end
    per = 2*(L+W);
    t_per = (per*Np)/ps;
    A = L*W*(id/100); s = nd*0.9;
    t_inf = (A/s)*kp/ps;
    NL = ceil(H/lt);
    t_mov = NL*tl;
    T = 1.2*(t_per+t_inf+t_mov)/60; % minutes
end

function Xoh = onehot_encode(X)
    fp = X(:,3); col = X(:,9);
    FP = onehot(fp,3); CL = onehot(col,7);
    Xoh = [X(:,1:2), FP, X(:,4:8), CL];
end

function Oh = onehot(v,K)
    Oh = zeros(numel(v),K);
    for i=1:numel(v), Oh(i,v(i))=1; end
end

function names = onehot_feat_names()
    names = [{'LayerThick','Infill%'}, ...
             arrayfun(@(k)sprintf('Pattern_%d',k),1:3,'UniformOutput',false), ...
             {'FilamentD','NozzleD','NozzleT','BedT','PrintSpeed'}, ...
             arrayfun(@(k)sprintf('Color_%d',k),1:7,'UniformOutput',false)];
end