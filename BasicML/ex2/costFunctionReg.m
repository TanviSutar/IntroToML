function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);
 
J = 0;
grad = zeros(size(theta));

theta_ = transpose(theta);
X_ = transpose(X);

[r,c] = size(X_);

sum = 0;
for i=1:m
    guess = 0;
    for j=1:r
        guess = guess + theta_(j)*X_(j,i);
    end
    sum = sum + ((-y(i)*log(sigmoid(guess))) - ((1-y(i))*log(1-sigmoid(guess))));
end

J = sum/m;

sum = 0;
for i=2:r
    sum = sum + theta_(i).^2;
end

J = J + (lambda/(2*m))*sum;

sum = 0;
for i=1:size(theta)
    sum = 0;
    for j=1:m
        guess = 0;
        for k=1:r
            guess = guess + theta_(k)*X_(k,j);
        end
        sum = sum + (sigmoid(guess)-y(j))*X_(i,j);
    end
    if(i==1)
        grad(i) = sum/m;
    else
       grad(i) = sum/m + (lambda/m)*theta_(i); 
    end
end

end
