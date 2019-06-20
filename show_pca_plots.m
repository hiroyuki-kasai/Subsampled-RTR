function  show_pca_plots()
    
    clc; 
    close all; 
    clear
    
    
    %% Select case
    %case_num = 1;   % Case P1
    %case_num = 2;   % Case P2
    case_num = 3;   % small instance
    
    if case_num == 1
        N = 5000000; d = 100; r = 5;
    elseif case_num == 2
        N = 500000; d = 1000; r = 5; 
    elseif case_num == 3
        N = 100000; d = 100; r = 5;         
    else
    end
    
    
    
    %% Define parameters
    maxepoch = 100;
    tolgradnorm = 1e-8;
    
    
    
    %% Generate data
    fprintf('generating data ... ');
    x_sample = randn(d, N);
    x_sample = diag(exprnd(2, d , 1))*x_sample;    
    x_sample = x_sample - repmat(mean(x_sample,2),1,size(x_sample,2));
    %cond(x_sample)
    fprintf('done.\n');

    % Iput data as cell
    data.x = mat2cell(x_sample, d, ones(N, 1)); %     
    
 
    
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

   

    %     % Consistency checks
    %     checkgradient(problem)
    %     pause;
    
    %     % Consistency checks
    %     checkhessian(problem)
    %     pause;
        
    
    
    
    %% Run algorithms    
    
    % Initialize
    Uinit = problem.M.rand();
 
    % Run RSD
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_sd, ~] = steepestdescent_mod(problem, Uinit, options); 
    
    % Run RCG
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_cg, ~] = conjugategradient_mod(problem, Uinit, options);  
    
    % Run RLBFGS
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;         
    [~, ~, infos_lbfgs] = lbfgs_mod(problem, Uinit, options); 
    
    
    % Run RSVRG
    clear options;    
    inner_repeat = 5;
    options.verbosity = 1;
    options.batchsize = floor(N/100);
    options.update_type = 'svrg';
    options.stepsize = 0.001;
    options.stepsize_type = 'fix';
    options.stepsize_lambda = 0;
    options.tolgradnorm = tolgradnorm; 
    options.boost = 0;
    options.svrg_type = 2; % effective only for R-SVRG variants
    options.maxinneriter = inner_repeat * N;
    options.transport = 'ret_vector';
    options.maxepoch = floor(maxepoch / (1 + inner_repeat)) * 2;
    [~, ~, infos_svrg, ~] = Riemannian_svrg(problem, Uinit, options);  
                
    
    % Run RTR (basic)
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;   
    options.samp_hess_init_size = N;      
    [~, ~, infos_tr, ~] = subsampled_rtr(problem, Uinit, options);       

    
    % Run Sub-sampled Hessian TR
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;     
    options.samp_hess_scheme = 'fix';
    options.samp_hess_init_size = floor(N/100);
    options.useExp = true;    
    [~, ~, infos_subHtr_fix, ~] = subsampled_rtr(problem, Uinit, options); 
    
    
    % Run Sub-sampled Hessian & Gradient TR
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;     
    options.samp_hess_scheme = 'fix';
    options.samp_hess_init_size = floor(N/100); 
    options.samp_grad_scheme = 'fix';
    options.samp_grad_init_size = floor(N/10);
    options.useExp = true;
    [~, ~, infos_subHGtr_fix, ~] = subsampled_rtr(problem, Uinit, options);     
    
 
    
    %% calculate optimality gap
    optgap_sd = abs([infos_sd.cost] - f_sol);
    optgap_cg = abs([infos_cg.cost] - f_sol);  
    optgap_lbfgs = abs([infos_lbfgs.cost] - f_sol); 
    optgap_svrg = abs([infos_svrg.cost] - f_sol);        
    optgap_tr = abs([infos_tr.cost] - f_sol);    
    optgap_subHtr_fix = abs([infos_subHtr_fix.cost] - f_sol); 
    optgap_subHGtr_fix = abs([infos_subHGtr_fix.cost] - f_sol);       

    
    
    %% Plots
    fs = 20;    
    line_color = {[255, 128, 0], [76, 153, 0], [255,0,255], [204, 204, 0], [0, 0, 255], [255, 0, 0], [255, 0, 0]};      

    titlestr = sprintf('PCA Problme: N=%d, d=%d, r=%d', N, d, r);   
    
    % Optimality gap (Train loss - optimum) vs. oracle calls     
    figure;
    semilogy([infos_sd.oraclecalls], optgap_sd,'-','LineWidth',2, 'Color', line_color{1}/255);                      hold on;
    semilogy([infos_cg.oraclecalls], optgap_cg, '-','LineWidth',2, 'Color', line_color{2}/255);                     hold on; 
    semilogy([infos_lbfgs.oraclecalls], optgap_lbfgs,'-','LineWidth',2, 'Color', line_color{3}/255);                hold on; 
    semilogy([infos_svrg.oraclecalls], optgap_svrg,'-','LineWidth',2, 'Color', line_color{4}/255);                  hold on;    
    semilogy([infos_tr.oraclecalls], optgap_tr,'-','LineWidth',2, 'Color', line_color{5}/255);                      hold on;
    semilogy([infos_subHtr_fix.oraclecalls], optgap_subHtr_fix,'-','LineWidth',2, 'Color', line_color{6}/255);      hold on;
    semilogy([infos_subHGtr_fix.oraclecalls], optgap_subHGtr_fix,'-.','LineWidth',2, 'Color', line_color{7}/255);   hold on;    
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Oracle calls','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Optimality gap','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('RSD', 'RCG', 'RLBFGS','RSVRG', 'RTR','Sub-H-RTR', 'Sub-HG-RTR');
    title(titlestr)
    
    
    % Optimality gap (Train loss - optimum) vs. processing time [sec]  
    figure;
    semilogy([infos_sd.time], optgap_sd,'-','LineWidth',2, 'Color', line_color{1}/255);                      hold on;
    semilogy([infos_cg.time], optgap_cg, '-','LineWidth',2, 'Color', line_color{2}/255);                     hold on; 
    semilogy([infos_lbfgs.time], optgap_lbfgs,'-','LineWidth',2, 'Color', line_color{3}/255);                hold on; 
    semilogy([infos_svrg.time], optgap_svrg,'-','LineWidth',2, 'Color', line_color{4}/255);                  hold on;    
    semilogy([infos_tr.time], optgap_tr,'-','LineWidth',2, 'Color', line_color{5}/255);                      hold on;
    semilogy([infos_subHtr_fix.time], optgap_subHtr_fix,'-','LineWidth',2, 'Color', line_color{6}/255);      hold on;
    semilogy([infos_subHGtr_fix.time], optgap_subHGtr_fix,'-.','LineWidth',2, 'Color', line_color{7}/255);   hold on;    
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time [sec]','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Optimality gap','FontName','Arial','FontSize',fs,'FontWeight','bold');
    legend('RSD', 'RCG', 'RLBFGS','RSVRG', 'RTR','Sub-H-RTR', 'Sub-HG-RTR');
    title(titlestr)    
 

end

