function hess = getPartialHessian(problem, x, d, I, I_p, storedb, key)
% Computes the Hessian of the cost function at x along d.
%
% function hess = getPartialHessian(problem, x, d, I, I_p)
% function hess = getPartialHessian(problem, x, d, I, I_p, storedb)
% function hess = getPartialHessian(problem, x, d, I, I_p, storedb, key)
%
% Returns the Hessian at x along d of the cost function described in the
% problem structure.
%
% Assume the cost function described in the problem structure is a sum of
% many terms, as
%
%    f(x) = sum_i f_i(x) for i = 1:d,

% where d is specified as d = problem.ncostterms.
% 
% For a subset I of 1:d, getPartialHessian obtains the Hessian of the
% partial cost function
% 
%    f_I(x) = sum_i f_i(x) for i = I.
%
% storedb is a StoreDB object, key is the StoreDB key to point x.
%
% If an exact Hessian is not provided, an approximate Hessian is returned
% if possible, without warning. If not possible, an exception will be
% thrown. To check whether an exact Hessian is available or not (typically
% to issue a warning if not), use canGetHessian.
%
% See also: getPrecon getApproxHessian canGetHessian

% This file is part of Manopt: www.manopt.org.
% Original author: Nicolas Boumal, Dec. 30, 2012.
% Contributors: 
% Change log: 
%
%   April 3, 2015 (NB):
%       Works with the new StoreDB class system.

    % Allow omission of the key, and even of storedb.
    if ~exist('key', 'var')
        if ~exist('storedb', 'var')
            storedb = StoreDB();
        end
        key = storedb.getNewKey();
    end
    
    
    if isfield(problem, 'partialhess')
    %% Compute the Hessian using hess.
	
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.partialhess)
            case 2
                hess = problem.partialhess(x, d);
            case 3
                % Obtain, pass along, and save the store for x.
                store = storedb.getWithShared(key);
                [hess, store] = problem.hess(x, d, store);
                storedb.setWithShared(store, key);
            case 4
                % Pass along the whole storedb (by reference), with key.
                hess = problem.partialhess(x, d, storedb, key);
            otherwise
                up = MException('manopt:getHessian:badhess', ...
                    'hess should accept 2, 3 or 4 inputs.');
                throw(up);
        end
    
    elseif isfield(problem, 'partialehess') && canGetEuclideanGradient(problem)
    %% Compute the Hessian using ehess.
    
        % We will need the Euclidean gradient for the conversion from the
        % Euclidean Hessian to the Riemannian Hessian.
        egrad = getEuclideanGradient(problem, x, storedb, key);
		
        % Check whether this function wants to deal with storedb or not.
        switch nargin(problem.partialehess)
            case 4
                ehess = problem.partialehess(x, d, I, I_p);
            case 5
                % Obtain, pass along, and save the store for x.
                store = storedb.getWithShared(key);
                [ehess, store] = problem.partialehess(x, d, I, I_p, store);
                storedb.setWithShared(store, key);
            case 6
                % Pass along the whole storedb (by reference), with key.
                ehess = problem.partialehess(x, d, I, I_p, storedb, key);
            otherwise
                up = MException('manopt:getPartialHessian:badehess', ...
                    'ehess should accept 4, 5 or 6 inputs.');
                throw(up);
        end
        
        % Convert to the Riemannian Hessian
        hess = problem.M.ehess2rhess(x, egrad, ehess, d);
        
    else
    %% Attempt the computation of an approximation of the Hessian.
        
        hess = getApproxHessian(problem, x, d, storedb, key);
        
    end
    
end
