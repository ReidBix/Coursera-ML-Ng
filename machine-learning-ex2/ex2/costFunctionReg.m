function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = size(X,2);
h = zeros(m,1);

for i = 1:m
    d = X(i,:) * theta;
    h(i) = 1.0 /(1.0 + exp(-d));
end    

theta0 = theta(2:n,:);
J = (-1.0 / m) * sum(y .* log(h) + (1 - y) .* log(1 - h)) + (lambda /(2.0 * m)) * (theta0' * theta0);

for j = 1:n
    grad(j) = (1.0/m)*sum((h-y)' * X(:,j));
    if j > 1
        grad(j) = grad(j) + (lambda/m) * theta(j);
    endif    
end    

% =============================================================

end
