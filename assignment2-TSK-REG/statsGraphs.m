function [RMSE, R2, NMSE, NDEI] = statsGraphs(id, valFIS, chkData, tstData, trnError, valError, mfPlots)

    % Evaluation function
    Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

    % Stats calculation
    Y = evalfis(tstData(:,1:end-1),valFIS);
    R2 = Rsq(Y, tstData(:,end));
    RMSE = sqrt(mse(Y,tstData(:,end)));
    NMSE = sum((Y - tstData(:,end)).^2) / sum((tstData(:,end) - mean(tstData(:,end))).^2);
    NDEI = sqrt(NMSE);
    
    % Graph Generation
    figure
    plot([trnError valError],'LineWidth',2); grid on;
    xlabel('# of Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    title(sprintf("TSK %d Training", id));
    figure
    hold on
    if mfPlots == 1
        for i=1:(size(chkData,2) - 1)
            subplot(2,3,i);
            plotmf(valFIS, 'input', i);
        end
    end
    suptitle(sprintf("TSK %d - Membership Functions", id));
    hold off
    
    figure
    histogram(abs(Y - tstData(:,end)));
    title(sprintf("TSK %d - Prediction Error", id));
    xlabel('Error');
    ylabel('Ν');
end