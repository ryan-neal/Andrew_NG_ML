function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
theta_length = size(theta);
grad = zeros(theta_length);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% get hypothesis function
hypothesis = sigmoid(X * theta);

% solve cost function
h1 = -y' * log(hypothesis);
h2 = (1-y)' * log(1-hypothesis);
% get cost function with regularization term for thetas > position 1
J = (1/m) * (h1 - h2) + ((lambda/(2*m)) * sum(theta(2:theta_length,1).^2));

% gradient descent - separate methods for first and all other thetas
% base gradient
error = hypothesis - y;
base_grad = (1/m) * (X' * error);

% theta 1
grad(1) = base_grad(1);

% theta 2 to the end of the array add a regularization term to the base
% gradient
grad(2:theta_length) = base_grad(2:theta_length) + ((lambda/m) * theta(2:theta_length,1));



% =============================================================

end
