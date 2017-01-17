function [c, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of c and sigma for Part 3 of the exercise
%where you select the optimal (c, sigma) learning parameters to use for SVM
%with RBF kernel
%   [c, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of c and 
%   sigma. You should complete this function to return the optimal c and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
c = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

err_final = 1;
vals = [0.01 0.03 0.1 0.3 1 3 10 30];
for i=1:8,
    for j=1:8,
        c_cur = vals(i);
        sig_cur = vals(j);
        train = svmTrain(X, y, c_cur, @(x1, x2) gaussianKernel(x1, x2, sig_cur));
        predictions = svmPredict(train, Xval);
        err = mean(double(predictions ~= yval));
        if err < err_final,
            c = c_cur;
            sigma = sig_cur;
            err_final = err;
        end
    end
end

% =========================================================================

end
