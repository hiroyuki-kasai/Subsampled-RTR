function  demo()
    
    clc; 
    close all; 
    clear
    

    %% Define parameters
    N = 10000; d = 100; r = 5;    
    maxepoch = 100;
    tolgradnorm = 1e-8;
    
    
    %% Generate data
    x_sample = randn(d, N);
    x_sample = diag(exprnd(2, d , 1))*x_sample;    
    x_sample = x_sample - repmat(mean(x_sample,2),1,size(x_sample,2));
    %cond(x_sample)

    % Iput data as cell
    data.x = mat2cell(x_sample, d, ones( N, 1)); %     
    
 
    
    %% Obtain solution
    coeff = pca(x_sample');
    x_star = coeff(:,1:r);
    f_sol = -0.5/N*norm(x_star'*x_sample, 'fro')^2;
    fprintf('f_sol: %.16e, cond = %.2f\n', f_sol, cond(x_sample));
    
    
    %% Set manifold
    problem.M = grassmannfactory(d, r);
    problem.ncostterms = N;
    problem.d = d;    
    problem.data = data;
    
    
    %% Define problem definitions
    problem.cost = @cost;
    function f = cost(U)
        f = -0.5*norm(U'*x_sample, 'fro')^2;
        f = f/N;
    end
    
    problem.egrad = @egrad;
    function g = egrad(U)
        g = - x_sample*(x_sample'*U);        
        g = g/N;
    end
    
    problem.partialegrad = @partialegrad;
    function g = partialegrad(U, indices)
        len = length(indices);
        x_sample_batchsize = x_sample(:,indices);        
        g = - x_sample_batchsize*(x_sample_batchsize'*U);
        g = g/len;
    end        

    problem.ehess = @ehess;
    function gdot = ehess(U, Udot)
        gdot = - x_sample*(x_sample'*Udot);
        gdot = gdot/N;
    end 

    problem.partialehess = @partialehess;
    function gdot = partialehess(U, Udot, indices, square_hess_diag)
        len = length(indices);

        x_sub_sample = x_sample(:, indices);
        gdot = - x_sub_sample * (x_sub_sample' * Udot);

        gdot = gdot/len;               
    end

  
    %% Run algorithms (Sub-H-RTR)   
    Uinit = problem.M.rand();
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;     
    options.samp_hess_scheme = 'fix';
    options.samp_hess_init_size = floor(N/100);
    options.useExp = true;    
    [~, ~, infos_subHtr_fix, ~] = subsampled_rtr(problem, Uinit, options); 
    optgap_subHtr_fix = abs([infos_subHtr_fix.cost] - f_sol); 

    
    %% Plots
    fs = 20;
    titlestr = sprintf('PCA Problme: N=%d, d=%d, r=%d', N, d, r);   
    
    % Optimality gap (Train loss - optimum) vs. oracle calls     
    figure;
    semilogy([infos_subHtr_fix.oraclecalls], optgap_subHtr_fix,'-r','LineWidth',2);
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Oracle calls','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Optimality gap','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('Sub-H-RTR');
    title(titlestr)
    
    
    % Optimality gap (Train loss - optimum) vs. processing time [sec]  
    figure;
    semilogy([infos_subHtr_fix.time], optgap_subHtr_fix,'-r','LineWidth',2); 
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time [sec]','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Optimality gap','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('Sub-H-RTR');    legend('Sub-H-RTR');
    title(titlestr)    
end

