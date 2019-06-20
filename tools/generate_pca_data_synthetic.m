function [x_sample, x_init, f_sol, error] = generate_pca_data_synthetic(N, d, r, generate_flag, path, calc_sol_flag)

    error = false;
    f_sol = 0;
    x_init = [];

    if generate_flag
    
        % generate data
        x_sample = randn(d, N);
        x_sample = diag(exprnd(2, d , 1))*x_sample;    
        x_sample = x_sample - repmat(mean(x_sample,2),1,size(x_sample,2));
        
    else
        input_filename = sprintf('%spca_samples_%d_d_%d_r_%d.mat', path, N, d, r);
        fprintf('\tloading %s ... ', input_filename);        
        input_data = load(input_filename);

        if input_data.d ~= d
            fprintf('generate_pca_data_synthetic: invalied dimension %d\n', input_data.d);
            error = true;
        end
        
        if input_data.r  ~= r
            fprintf('generate_pca_data_synthetic: invalid rank %d\n', nput_data.r);
            error = true;
        end
        
        data_x_samples = input_data.x_sample;
        x_sample = data_x_samples{1};
        x_init = input_data.x_init{1};
        f_sol = input_data.f_star{1};
        calc_sol_flag = false;
        fprintf('done.\n');        
    end
    
    % calcualte solution    
    if calc_sol_flag
        fprintf('calculating a solution of PCA .....\n');         
        coeff = pca(x_sample');
        x_star = coeff(:,1:r);
        f_sol = -0.5/N*norm(x_star'*x_sample, 'fro')^2;
        fprintf('f_sol: %.16e\n', f_sol);        
    end

end

