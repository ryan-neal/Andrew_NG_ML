function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
[theta_row, theta_col] = size(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% get error
error = (X * theta) - y;

% create matrix of squared errors
J = J + error.^2;
J = J/(2 * m);

% sum of matrix is cost function
J = sum(J);

% regularization
reg_term = sum(((theta(2:theta_row, 1).^2) * (lambda/(2*m))));

% add regularization for final cost function
J = J + reg_term;

% calculate base gradient
base_grad = (X' * error) * (1/m);
grad(1) = base_grad(1);

% gradient regularization
grad_reg = (lambda/m) * theta(2:theta_row, 1);

% add regularization to every gradient, but the first
grad(2:theta_row) = base_grad(2:theta_row) + grad_reg;





% =========================================================================

grad = grad(:);

end
