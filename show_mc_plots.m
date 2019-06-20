function show_mc_plots()

    clc; 
    close all; 
    clear
    
    
    %% Select case
    %case_num = 1;   % Case M1
    %case_num = 2;   % Case M2
    case_num = 3;   % small instance
    
    
    
    if case_num == 1
        N = 100000; d = 100; r = 5;
        condition_number = 5;
    elseif case_num == 2
        N = 100000; d = 100; r = 5;
        condition_number = 20;
    elseif case_num == 3
        N = 10000; d = 30; r = 5;
        condition_number = 20;        
    else
    end 
    
    

    %% Define parameters
    maxepoch = 100;
    tolgradnorm = 1e-8;    
    over_sampling = 4;
    noiseFac = 1e-10;
    %sqn_mem_size = 0;
    
    NumEntries_train = over_sampling*r*(N + d -r);      
    NumEntries_test = over_sampling*r*(N + d -r);      

    
    %% Generate data
    fprintf('generating data ... \n');    
    % Generate well-conditioned or ill-conditioned data
    M = over_sampling*r*(N + d -r); % total entries
    
    % The left and right factors which make up our true data matrix Y.
    YL = randn(d, r);
    YR = randn(N, r);
    
    % Condition number
    if condition_number > 0
        YLQ = orth(YL);
        YRQ = orth(YR);
        
        s1 = 1000;
        %     step = 1000; S0 = diag([s1:step:s1+(r-1)*step]*1); % Linear decay
        S0 = s1*diag(logspace(-log10(condition_number),0,r)); % Exponential decay
        
        YL = YLQ*S0;
        YR = YRQ;
        
        fprintf('Creating a matrix with singular values...\n')
        for kk = 1: length(diag(S0))
            fprintf('%s \n', num2str(S0(kk, kk), '%10.5e') );
        end
        singular_vals = svd(YL'*YL);
        condition_number = sqrt(max(singular_vals)/min(singular_vals));
        fprintf('Condition number is %f \n', condition_number);
    end
    cn = floor(condition_number);
    
    % Select a random set of M entries of Y = YL YR'.
    idx = unique(ceil(N*d*rand(1,(10*M))));
    idx = idx(randperm(length(idx)));
    
    [I, J] = ind2sub([d, N],idx(1:M));
    [J, inxs] = sort(J); I=I(inxs)';
    
    % Values of Y at the locations indexed by I and J.
    S = sum(YL(I,:).*YR(J,:), 2);
    S_noiseFree = S;
    
    % Add noise.
    noise = noiseFac*max(S)*randn(size(S));
    S = S + noise;
    
    values = sparse(I, J, S, d, N);
    indicator = sparse(I, J, 1, d, N);
    

    % Creat the cells
    samples(N).colnumber = []; % Preallocate memory.
    for k = 1 : N
        % Pull out the relevant indices and revealed entries for this column
        idx = find(indicator(:, k)); % find known row indices
        values_col = values(idx, k); % the non-zero entries of the column
        
        samples(k).indicator = idx;
        samples(k).values = values_col;
        samples(k).colnumber = k;
    end 
    
    % Test data
    idx_test = unique(ceil(N*d*rand(1,(10*M))));
    idx_test = idx_test(randperm(length(idx_test)));
    [I_test, J_test] = ind2sub([d, N],idx_test(1:M));
    [J_test, inxs] = sort(J_test); I_test=I_test(inxs)';
    
    % Values of Y at the locations indexed by I and J.
    S_test = sum(YL(I_test,:).*YR(J_test,:), 2);
    values_test = sparse(I_test, J_test, S_test, d, N);
    indicator_test = sparse(I_test, J_test, 1, d, N);
    
    samples_test(N).colnumber = [];
    for k = 1 : N
        % Pull out the relevant indices and revealed entries for this column
        idx = find(indicator_test(:, k)); % find known row indices
        values_col = values_test(idx, k); % the non-zero entries of the column
        
        samples_test(k).indicator = idx;
        samples_test(k).values = values_col;
        samples_test(k).colnumber = k;
    end
    
    % for grouse
    data_ls.rows = I;
    data_ls.cols = J';
    data_ls.entries = S;
    data_ls.nentries = length(data_ls.entries);

    data_test.rows = I_test;
    data_test.cols = J_test';
    data_test.entries = S_test;
    data_test.nentries = length(data_test.entries);        
    fprintf('done.\n');    
    
    
    %% Set manifold
    problem.M = grassmannfactory(d, r);
    problem.ncostterms = N;
    problem.d = d;    
   
    % Define problem definitions
    problem.cost = @cost;
    function f = cost(U)
        W = mylsqfit(U, samples);
        f = 0.5*norm(indicator.*(U*W') - values, 'fro')^2;
        f = f/N;
    end
    
    problem.egrad = @egrad;
    function g = egrad(U)
        W = mylsqfit(U, samples);
        g = (indicator.*(U*W') - values)*W;
        g = g/N;
    end

    problem.ehess = @ehess;
    function gdot = ehess(U, Udot) 
        [W, Wdot] = mylsqfitdot(U, Udot, samples);
        gdot = (indicator.*(U*W') - values)*Wdot +  (indicator.*(U*Wdot' + Udot*W'))*W;
        gdot = gdot/N;
    end
    
    
    problem.partialegrad = @partialegrad;
    function g = partialegrad(U, idx_batch)
        g = zeros(d, r);
        m_batchsize = length(idx_batch);
        for ii = 1 : m_batchsize
            colnum = idx_batch(ii);
            w = mylsqfit(U, samples(colnum));
            indicator_vec = indicator(:, colnum);
            values_vec = values(:, colnum);
            g = g + (indicator_vec.*(U*w') - values_vec)*w;
        end
        g = g/m_batchsize;
    end

   
    problem.partialehess = @partialehess;
	function gdot = partialehess(U, Udot, idx_batch, square_hess_diag) % We need Udot and the idx_batch.
      if 0
            gdot = zeros(d, r);
            m_batchsize = length(idx_batch);
            for ii = 1 : m_batchsize
                colnum = idx_batch(ii);
                [w, wdot] = mylsqfitdot(U, Udot, samples(colnum)); % we compute both w and wdot. This has some redundant computations because w is already computed in partialegrad
                indicator_vec = indicator(:, colnum);
                values_vec = values(:, colnum);
                gdot = gdot + (indicator_vec.*(U*w') - values_vec)*wdot ... % we need both w and wdot. w is obtained from partialegrad, but we compute here again.
                    + (indicator_vec.*(U*wdot' + Udot*w'))*w;
            end
            gdot = gdot/m_batchsize;

        else
            m_batchsize = length(idx_batch);

            sub_samples = samples(idx_batch);
            sub_values = values(:, idx_batch);
            sub_indicator = indicator(:, idx_batch);

            [W, Wdot] = mylsqfitdot(U, Udot, sub_samples);
            gdot = (sub_indicator.*(U*W') - sub_values)*Wdot +  (sub_indicator.*(U*Wdot' + Udot*W'))*W;
            gdot = gdot/m_batchsize;        
        end
    end

    
    function stats = mc_mystatsfun(problem, U, stats)
        W = mylsqfit(U, samples_test);
        f_test = 0.5*norm(indicator_test.*(U*W') - values_test, 'fro')^2;
        f_test = f_test/N;
        stats.cost_test = f_test;
    end


    function W = mylsqfit(U, currentsamples)
        W = zeros(length(currentsamples), size(U, 2));
        for ii = 1 : length(currentsamples)
            % Pull out the relevant indices and revealed entries for this column
            IDX = currentsamples(ii).indicator;
            values_Omega = currentsamples(ii).values;
            U_Omega = U(IDX,:);
            
            % Solve a simple least squares problem to populate W.
            %OmegaUtUOmega = U_Omega'*U_Omega;
            OmegaUtUOmega = U_Omega'*U_Omega + 1e-10*eye(r);            
            W(ii,:) = (OmegaUtUOmega\(U_Omega'*values_Omega))';

        end
    end

    
    function [W, Wdot] = mylsqfitdot(U, Udot, currentsamples)
        W = zeros(length(currentsamples), size(U, 2));
        Wdot = zeros(size(W));
        for ii = 1 : length(currentsamples)
            % Pull out the relevant indices and revealed entries for this column
            IDX = currentsamples(ii).indicator;
            values_Omega = currentsamples(ii).values;
            U_Omega = U(IDX,:);
            Udot_Omega = Udot(IDX,:);
            
            % Solve a simple least squares problem to populate W and Wdot
            OmegaUtUOmega = U_Omega'*U_Omega;
            W(ii,:) = (OmegaUtUOmega\(U_Omega'*values_Omega))';

            UOmegaW = U_Omega*(W(ii,:))';
            UdotOmegaW = Udot_Omega*(W(ii,:))';

            Wdot(ii,:) = (OmegaUtUOmega\(Udot_Omega'*values_Omega - U_Omega'* UdotOmegaW  - Udot_Omega'* UOmegaW ))';
        end
    end   

    
    % Consistency checks
%     checkgradient(problem)
%     pause;
%     
%     
%     checkhessian(problem);
%     pause;
    
    
%% Run algorithms    
    % Initialize
    x_init = problem.M.rand();
    
    % Run RSD
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @mc_mystatsfun;                
    [~, ~, infos_sd, ~] = steepestdescent_mod(problem, x_init, options);       
    
    
    % Run RCG
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @mc_mystatsfun;                
    [~, ~, infos_cg, ~] = conjugategradient_mod(problem, x_init, options);   
    
    
    % Run RLBFGS
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm; 
    options.statsfun = @mc_mystatsfun;     
    [~, ~, infos_lbfgs] = lbfgs_mod(problem, x_init, options); 
    
    
    % Run RSVRG
    clear options;    
    inner_repeat = 5;
    options.verbosity = 1;
    options.batchsize = floor(N/100);
    options.update_type = 'svrg';
    options.stepsize = 0.01;
    options.stepsize_type = 'fix';
    options.stepsize_lambda = 0;
    options.tolgradnorm = tolgradnorm; 
    options.boost = 0;
    options.svrg_type = 2; % effective only for R-SVRG variants
    options.maxinneriter = inner_repeat * N;
    options.transport = 'ret_vector';
    options.maxepoch = floor(maxepoch / (1 + inner_repeat)) * 2;
    options.statsfun = @mc_mystatsfun;       
    [~, ~, infos_svrg, ~] = Riemannian_svrg(problem, x_init, options);      
    
    
    % Run RTRMC
    %clear options;
    %options.maxiterations = maxepoch;
    %[infos_rtrmc] = rtrmc_rapper(x_init, d, N, r, data_ls, data_test, options);   
                      
    
    % Run RTR
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @mc_mystatsfun;                
    [~, ~, infos_tr, ~] = subsampled_rtr(problem, x_init, options);         
    
    
    % Run Sub-sampled Hessian TR
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm;
    options.statsfun = @mc_mystatsfun;     
    options.samp_hess_scheme = 'fix';
    options.samp_hess_init_size = floor(N/100);
    options.samp_scheme = 'uniform';
    [~, ~, infos_subHtr_fix, ~] = subsampled_rtr(problem, x_init, options);    
    
    
    % Run Sub-sampled Hessian & Gradient TR
    clear options;
    options.maxiter = maxepoch;
    options.tolgradnorm = tolgradnorm; 
    options.statsfun = @mc_mystatsfun;         
    options.samp_hess_scheme = 'fix';
    options.samp_hess_init_size = floor(N/100); 
    options.samp_grad_scheme = 'fix';
    options.samp_grad_init_size = floor(N/10);       
    [~, ~, infos_subHGtr_fix, ~] = subsampled_rtr(problem, x_init, options);     
    
    

    %% Plots
    fs = 20;    
    line_color = {[255, 128, 0], [76, 153, 0], [255,0,255], [204, 204, 0], [153,76,0], [0, 0, 255], [255, 0, 0], [255, 0, 0]};      
    titlestr = sprintf('MC Problme: N=%d, d=%d, r=%d, cn=%d', N, d, r, cn);   
    

    % Train MSE vs. oracle calls 
    figure;
    semilogy([infos_sd.oraclecalls], [infos_sd.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{1}/255);                    hold on;
    semilogy([infos_cg.oraclecalls], [infos_cg.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{2}/255);                    hold on;  
    semilogy([infos_lbfgs.oraclecalls], [infos_lbfgs.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{3}/255);              hold on;
    semilogy([infos_svrg.oraclecalls], [infos_svrg.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{4}/255);                hold on;
    %semilogy([infos_rtrmc.oraclecalls], [infos_rtrmc.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{5}/255);              hold on;    
    semilogy([infos_tr.oraclecalls], [infos_tr.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{6}/255);                    hold on;
    semilogy([infos_subHtr_fix.oraclecalls], [infos_subHtr_fix.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{7}/255);    hold on;
    semilogy([infos_subHGtr_fix.oraclecalls], [infos_subHGtr_fix.cost] * 2 * N / NumEntries_train,'-.','LineWidth',2, 'Color', line_color{8}/255); hold on;    
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Oracle calls','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Means square error on train set \Gamma','FontName','Arial','FontSize',fs,'FontWeight','bold');
    %legend('RSD', 'RCG', 'RLBFGS','RSVRG','RTRMC','RTR','Sub-H-RTR', 'Sub-HG-RTR');
    legend('RSD', 'RCG', 'RLBFGS','RSVRG','RTR','Sub-H-RTR', 'Sub-HG-RTR');
    title(titlestr)    

    % Train MSE vs. processing time [sec]     
    figure;
    semilogy([infos_sd.time], [infos_sd.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{1}/255);                    hold on;
    semilogy([infos_cg.time], [infos_cg.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{2}/255);                    hold on;  
    semilogy([infos_lbfgs.time], [infos_lbfgs.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{3}/255);              hold on;
    semilogy([infos_svrg.time], [infos_svrg.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{4}/255);                hold on;
    %semilogy([infos_rtrmc.time], [infos_rtrmc.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{5}/255);              hold on;    
    semilogy([infos_tr.time], [infos_tr.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{6}/255);                    hold on;
    semilogy([infos_subHtr_fix.time], [infos_subHtr_fix.cost] * 2 * N / NumEntries_train,'-','LineWidth',2, 'Color', line_color{7}/255);    hold on;
    semilogy([infos_subHGtr_fix.time], [infos_subHGtr_fix.cost] * 2 * N / NumEntries_train,'-.','LineWidth',2, 'Color', line_color{8}/255); hold on;    
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Means square error on train set \Gamma','FontName','Arial','FontSize',fs,'FontWeight','bold');
    %legend('RSD', 'RCG', 'RLBFGS','RSVRG','RTRMC','RTR','Sub-H-RTR', 'Sub-HG-RTR');
    legend('RSD', 'RCG', 'RLBFGS','RSVRG','RTR','Sub-H-RTR', 'Sub-HG-RTR');
    title(titlestr)  
    
    
    % Test MSE vs. oracle calls 
    figure;
    semilogy([infos_sd.oraclecalls], [infos_sd.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{1}/255);                    hold on;
    semilogy([infos_cg.oraclecalls], [infos_cg.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{2}/255);                    hold on;  
    semilogy([infos_lbfgs.oraclecalls], [infos_lbfgs.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{3}/255);              hold on;
    semilogy([infos_svrg.oraclecalls], [infos_svrg.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{4}/255);                hold on;
    %semilogy([infos_rtrmc.oraclecalls], [infos_rtrmc.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{5}/255);              hold on;    
    semilogy([infos_tr.oraclecalls], [infos_tr.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{6}/255);                    hold on;
    semilogy([infos_subHtr_fix.oraclecalls], [infos_subHtr_fix.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{7}/255);    hold on;
    semilogy([infos_subHGtr_fix.oraclecalls], [infos_subHGtr_fix.cost_test] * 2 * N / NumEntries_test,'-.','LineWidth',2, 'Color', line_color{8}/255); hold on;    
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Oracle calls','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Means square error on test set','FontName','Arial','FontSize',fs,'FontWeight','bold');
    %legend('RSD', 'RCG', 'RLBFGS','RSVRG','RTRMC','RTR','Sub-H-RTR', 'Sub-HG-RTR');
    legend('RSD', 'RCG', 'RLBFGS','RSVRG','RTR','Sub-H-RTR', 'Sub-HG-RTR');
    title(titlestr)    

    % Test MSE vs. processing time [sec]     
    figure;
    semilogy([infos_sd.time], [infos_sd.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{1}/255);                    hold on;
    semilogy([infos_cg.time], [infos_cg.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{2}/255);                    hold on;  
    semilogy([infos_lbfgs.time], [infos_lbfgs.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{3}/255);              hold on;
    semilogy([infos_svrg.time], [infos_svrg.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{4}/255);                hold on;
    %semilogy([infos_rtrmc.time], [infos_rtrmc.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{5}/255);              hold on;    
    semilogy([infos_tr.time], [infos_tr.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{6}/255);                    hold on;
    semilogy([infos_subHtr_fix.time], [infos_subHtr_fix.cost_test] * 2 * N / NumEntries_test,'-','LineWidth',2, 'Color', line_color{7}/255);    hold on;
    semilogy([infos_subHGtr_fix.time], [infos_subHGtr_fix.cost_test] * 2 * N / NumEntries_test,'-.','LineWidth',2, 'Color', line_color{8}/255); hold on;    
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel(ax1,'Means square error on test set','FontName','Arial','FontSize',fs,'FontWeight','bold');
    %legend('RSD', 'RCG', 'RLBFGS','RSVRG','RTRMC','RTR','Sub-H-RTR', 'Sub-HG-RTR');
    legend('RSD', 'RCG', 'RLBFGS','RSVRG','RTR','Sub-H-RTR', 'Sub-HG-RTR');
    title(titlestr)      
    
    
    
    
    
  
end

