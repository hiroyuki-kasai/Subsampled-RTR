function stats = rtrmc_rapper(Xinit, d, N, r, data_ls, data_test, option)

    %% Feeding the problem instance to RTRMC

    % Pick a value for lambda, the regularization parameter
    lambda = 1e-5;


    % Randomize the data order
    I = data_ls.rows;
    J = data_ls.cols;
    X = data_ls.entries;
    C = ones(size(X));

    % Build a problem structure
    problem = buildproblem(I, J, X, C, d, N, r, lambda);
    
    
    if ~isempty(data_test)
        % Randomize the data order
        I_test = data_test.rows;
        J_test = data_test.cols;
        X_test = data_test.entries;
        C_test = ones(size(X_test));
        
        % Build a problem structure
        problem_test = buildproblem(I_test, J_test, X_test, C_test, d, N, r, lambda);
    else
        problem_test = [];
    end
    


    % Compute an initial guess (this-SVD based)
    %initstart = tic;
    %U0 = initialguess(problem);
    %inittime = toc(initstart);

    % [Optional] If we want to track the evolution of the RMSE as RTRMC
    % iterates, we can do so by specifying the exact solution in factored form
    % in the problem structure and asking RTRMC to compute the RMSE in the
    % options structure. See the subfunction computeRMSE in rtrmc.m to see how
    % to compute the RMSE on a test set if the whole matrix is not available in
    % a factorized A*B form (which is typical in actual applications).
    %problem.A = A;
    %problem.B = B;

    % Setup the options for the RTRMC algorithm.
    % These are the algorithms shown in the papers:
    %  RTRMC 2  : method = 'rtr', order = 2, precon = false
    %  RTRMC 2p : method = 'rtr', order = 2, precon = true
    %  RTRMC 1  : method = 'rtr', order = 1, precon = false
    %  RCGMC    : method = 'cg',  order = 1, precon = false
    %  RCGMCp   : method = 'cg',  order = 1, precon = true
    opts.method = 'rtr';     % 'cg' or 'rtr', to choose the optimization algorithm
    opts.order = 2;          % for rtr only: 2 if Hessian can be used, 1 otherwise
    opts.precon = false;      % with or without preconditioner
    opts.maxiter = option.maxiterations;      % stopping criterion on the number of iterations
    opts.maxinner = 50;      % for rtr only : maximum number of inner iterations
    opts.tolgradnorm = 1e-8; % stopping criterion on the norm of the gradient
    opts.verbosity = 2;      % how much information to display during iterations
    opts.computeRMSE = true; % set to true if RMSE is to be computed at each step
    opts.use_default_update_alg = true;    


    % Call the algorithm here. The outputs are U and W such that U is
    % orthonormal and the product U*W is the matrix estimation. The third
    % output, stats, is a structure array containing lots of information about
    % the iterations made by the optimization algorithm. In particular, timing
    % information should be retrieved from stats, as in the code below.
    [U, W, stats] = rtrmc_mod(problem, problem_test, opts, Xinit);



%     time = inittime + [stats.time];
%     rmse = [stats.RMSE];
% 
%     switch opts.method
%         case 'rtr'
%             semilogy(time, rmse, '.-');
%             title('Running Riemannian trust-regions');
%         case 'cg'
%             semilogy(time, rmse, '.-');
%             S = [stats.linesearch];
%             ls_evals = [S.costevals];
%             for i = 1 : length(ls_evals)
%                 text(time(i), rmse(i)*1.20, num2str(ls_evals(i)));
%             end
%             title(sprintf('Running Riemannian conjugate gradients.\nNumbers indicate line-search cost evaluations.'));
%     end
%     xlabel('Time [s]');
%     ylabel('RMSE');
% 
% 
%     cd('/output/');
% 
%     % plot to file
%     saveas(gcf, 'graph.png', 'png');
%     
%     % plot to interactive plotly graph
%     fig = plotlyfig(gcf, 'filename', 'graph');
%     fig.strip;
%     fig.layout.yaxis1.exponentformat = 'power';
%     plotlyoffline(fig); % save figure to HTML file

end
