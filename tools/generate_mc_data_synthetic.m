function [samples, values, indicator, samples_test, values_test, indicator_test, data_ls, data_test] = generate_mc_data_synthetic(N, d, r, condition_number, over_sampling, noiseFac)
        %% Generate data
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
            for kk = 1: length(diag(S0));
                fprintf('%s \n', num2str(S0(kk, kk), '%10.5e') );
            end
            singular_vals = svd(YL'*YL);
            condition_number = sqrt(max(singular_vals)/min(singular_vals));
            fprintf('Condition number is %f \n', condition_number);
        end

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
        for k = 1 : N,
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
        for k = 1 : N,
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

end

