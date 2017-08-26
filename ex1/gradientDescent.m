function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    S = 0;
    h = X * theta;
     for i=1:2
       S = sum((h-y).*X(:,i));
       theta(i, 1) = theta(i, 1) - alpha/m*S;
     end  
    J_history(iter,1) = computeCost(X, y, theta);
end
end
