function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

X_ = transpose(X);
theta_ = transpose(theta);

sum = 0;
for i = 1:length(X_(1,:))
    guess = 0;
    for j = 1:length(theta_)
        guess = guess + theta_(j)*X_(j,i);
    end
    sum = sum + (guess - y(i))^2;
end

J = (sum)/(2*m);


% =========================================================================

end
