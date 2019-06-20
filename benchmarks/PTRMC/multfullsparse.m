function A = multfullsparse(B, C, mask)
% FUNCTION A = MULTFULLSPARSE(B, C, MASK)
%
% Efficient multiplication of a full and a sparse matrix. This code relies
% on an ugly C-mex trick, and should not be used for anything else than
% what it was written for.
%
% B is a full p-by-m matrix.
% D is a sparse m-by-n matrix with entries C at positions specified by the
%        sparse mask matrix. Please note that the entries in mask *will*
%        be modified by this function: it is a dummy place holder.
% A is the full p-by-n matrix BD.
% Complexity: O( p*nnz(D) ).
%
% Nicolas Boumal, UCLouvain, May 20, 2011.
% http://perso.uclouvain.be/nicolas.boumal/RTRMC/
%
% SEE ALSO: multsparsefull

    setsparseentries(mask, C);
    A = B*mask;

end
