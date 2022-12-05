%% INIT
clear
clc

%% Data Loading and Preprocessing

data = load('airfoil_self_noise.dat');
epoch_num = 300;
performanceMetrics = zeros(4,4);
preProcessMethod = 1;
[trnData, chkData, tstData] = split_scale(data, preProcessMethod);

%% Define TSK Models

% We are using genfis with the grid partition option 
% instead of the genfis1 function due to deprecation in 
% the newer matlab version that is used.
% https://www.mathworks.com/help/fuzzy/genfis1.html#mw_9fa3b177-55f0-40ff-a58f-04c63cb5ae13

trnIn = trnData(:, 1:(end-1));
trnOut = trnData(:, end);
model_names = {'TSK-1','TSK-2','TSK-3','TSK-4'};
metrics = {'RMSE','R2','NMSE','NDEI'};
inputMFtype = 'gbellmf';
outputMFnumber = [2,3,2,3];
outputMFtype = {'constant','constant','linear','linear'};

% models are saved so the workspace can be reproduced
models = cell(4);
errors = cell(4);
val_models = cell(4);
val_errors = cell(4);

for tsk_n=1:4
    % genfis options
    opt = genfisOptions('GridPartition');
    opt.NumMembershipFunctions = outputMFnumber(tsk_n);
    opt.InputMembershipFunctionType = inputMFtype;
    opt.OutputMembershipFunctionType = outputMFtype{tsk_n};
    % Train Models
    fis = genfis(trnIn,trnOut,opt);
    anfis_opt = anfisOptions('InitialFIS', fis, ...
                'EpochNumber', epoch_num, ...
                'ErrorGoal', 0, ...
                'InitialStepSize', 0.01, ...
                'StepSizeDecreaseRate', 0.9, ...
                'StepSizeIncreaseRate', 1.1, ...
                'ValidationData', chkData, ...
                'OptimizationMethod', 1);
            
    [tsk_trnFIS, tsk_trnError, ~, tsk_valFIS, tsk_valError]=anfis(trnData, anfis_opt);
    
    % store for reusability if needed
    models{tsk_n} = tsk_trnFIS;
    errors{tsk_n} = tsk_trnError;
    val_models{tsk_n} = tsk_valFIS;
    val_errors{tsk_n} = tsk_valError;
    [RMSE, R2, NMSE, NDEI] = statsGraphs(tsk_n, tsk_valFIS, chkData, tstData, tsk_trnError, tsk_valError, 1);
    performanceMetrics(tsk_n,:) = [RMSE, R2, NMSE, NDEI];
    
end

performanceMetrics = array2table(performanceMetrics,'VariableNames',metrics,'RowNames',model_names);
