%% INIT
clear
clc

%% Data Loading and Preprocessing
data = csvread("train.csv", 1, 0);
preProcessMethod = 1;
[trnData, chkData, tstData] = split_scale(data, preProcessMethod);

%% Grid Search

% Calculate characteristics potential
[ranks,weights] = relieff(data(:, 1:(end-1)), data(:, end), 10);

% Performance Matrix Indices
x = 0;  y = 0;

characteristics_range = 4:2:16;
cluster_r_range = 0.2:0.2:1.0;

stats = zeros(length(characteristics_range), length(cluster_r_range));

% Select number of characteristics
for n_features = characteristics_range
    
    x = x + 1;
    y = 0;
    
    % Select cluster radius
    for r_cluster = cluster_r_range
        
        y = y + 1;
        
        fprintf("START 5FOLD CROSSVAL FOR R=%f and N=%f", r_cluster, n_features);
        
        % 5-fold Cross Validation Data partitioning
        idx = randperm(length(data));
        idxLen5th = round(length(idx)*0.2);
        
        % Cross Validation
        for i = 0:4
            
            % 5-fold Cross Validation Data partitioning
            cvTrnDataIdx = idx;
            if i < 4
                cvChkDataIdx = idx(1+i*idxLen5th:1+(i+1)*idxLen5th);
                cvTrnDataIdx(1+i*idxLen5th:1+(i+1)*idxLen5th) = [];
            else
                cvChkDataIdx = idx(1+i*idxLen5th:end);
                cvTrnDataIdx(1+i*idxLen5th:end) = [];
            end
            
            %82 for the results data column
            cvTrnData = data(cvTrnDataIdx, [ranks(1:n_features) 82]);
            cvChkData = data(cvChkDataIdx, [ranks(1:n_features) 82]);
            cvTrnDataIn = cvTrnData(:, 1:(end-1));
            cvTrnDataOut = cvTrnData(:, end);
            
            % TSK Model creation and training
            opt = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', r_cluster);
            fis = genfis(cvTrnDataIn, cvTrnDataOut, opt);
            anfis_opt = anfisOptions('InitialFIS', fis, ...
                            'EpochNumber', 100, ...
                            'ErrorGoal', 0, ...
                            'InitialStepSize', 0.01, ...
                            'StepSizeDecreaseRate', 0.9, ...
                            'StepSizeIncreaseRate', 1.1, ...
                            'ValidationData', cvChkData, ...
                            'OptimizationMethod', 1);
            
            % Train Model
            [~, ~, ~, ~, valError] = anfis(cvTrnData, anfis_opt);
                        
            % Record error stats
            stats(x,y) = stats(x,y) + min(valError)/5;
        end
    end
end

%% Optimal Configuration Selection
[idx_n_features, idx_r_cluster] = find(stats == min(stats(:)));
optChar = characteristics_range(idx_n_features);
optRadius = cluster_r_range(idx_r_cluster);

%% Select final features in data
trnData = trnData(:, [ranks(1:optChar) 82]);
chkData = chkData(:, [ranks(1:optChar) 82]);
tstData = tstData(:, [ranks(1:optChar) 82]);

%% Define Final Model

fprintf("TRAINING FINAL MODEL: R = %f, N=%f", optRadius, optChar);

trnDataIn = trnData(:, 1:(end-1));
trnDataOut = trnData(:, end);


opt = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', optRadius);
fis = genfis(trnDataIn, trnDataOut, opt);
anfis_opt = anfisOptions('InitialFIS', fis, ...
                'EpochNumber', 350, ...
                'ErrorGoal', 0, ...
                'InitialStepSize', 0.01, ...
                'StepSizeDecreaseRate', 0.9, ...
                'StepSizeIncreaseRate', 1.1, ...
                'ValidationData', chkData, ...
                'OptimizationMethod', 1);

%% Train Model
[trnFIS, trnError, ~, valFIS, valError] = anfis(trnData, anfis_opt);

%% Benchmarking and Graphing

% Stats, Learning Curves, Final MFs
[RMSE, R2, NMSE, NDEI] = statsGraphsCV(1, valFIS, chkData, tstData, trnError, valError, 1);

% Initial MFs
figure
for i=1:(size(chkData,2) - 1)
    subplot(4,4,i);
    plotmf(fis, 'input', i);
end
suptitle('Initial Membership Functions');

% Predictions - Data Plot
Y = evalfis(tstData(:,1:end-1),valFIS);
tstY = [tstData(:, end) Y];
tstY = sortrows(tstY);
figure
plot(tstY(:,1), '-b', 'LineWidth', 2.5)
hold on
plot(tstY(:,2), '.r')
title('Predictions vs Actual Data');
ylabel('Value');
xlabel('Dataset Entry');

% Grid Search Error Surface
X = (cluster_r_range);
Y = (characteristics_range);
surf(X, Y, stats);
xlabel('Cluster Influence Range')
ylabel('Number of Features')
zlabel('RMSE')
title('Grid Search Error')

metrics = {'RMSE','R2','NMSE','NDEI'};
performanceMetrics = array2table([RMSE, R2, NMSE, NDEI],'VariableNames',metrics,'RowNames',{'Final Model'});