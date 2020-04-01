function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y); 
J_history = zeros(num_iters, 1);

X_ = transpose(X);
theta_ = transpose(theta);
th = theta_;

for iter = 1:num_iters

   for j = 1:length(theta)
        sum = 0;
        for i=1:m
            sum1 = 0;
            for k=1:length(theta)
                sum1 = sum1 + theta_(k)*X_(k,i);
            end
            sum = sum + ((sum1-y(i))*X_(j,i));
        end
        th(j) = th(j) - (alpha*sum)/m;
    end
    
    theta_ = th;
    theta = transpose(theta_);
    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
