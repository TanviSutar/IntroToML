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

J = 1/(2*m)*sum(((X*theta)-y).^2) + lambda/(2*m)*sum(theta(2:end,1).^2);%squared mean error

theta_ = [zeros(size(theta(1,:))) ;theta(2:end,:)*(lambda/m)];%adding regularisation term 

grad = theta_ + (1/m)*transpose((transpose((X*theta)-y))*X);

grad = grad(:);

end
