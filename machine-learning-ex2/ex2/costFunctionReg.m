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

n = length(theta); % number of features

temp = 0;
temp2 = 0;
for i=1:m
	temp = temp + (-y(i) * log (1 ./ (1 + (e .^ (- X(i,:) * theta)))) ...
	- (1 - y(i)) * log (1 - (1 ./ (1 + (e .^ (- X(i,:) * theta))))));
	end;
for j=2:n
	temp2 = temp2 + theta(j)^2;
J = (temp / m) + ((lambda / (2 * m)) * temp2);

temp3 = zeros(size(X, 2), 1);

for i=1:m
	temp3(1) = temp3(1) + (sigmoid(X(i,:) * theta) - y(i));
	end;
grad(1) = temp3(1) / m;

for j = 2:n
	for i=1:m
		temp3(j) = temp3(j) + ((sigmoid(X(i,:) * theta) - y(i)) * X(i,j));
		end;
	grad(j) = (temp3(j) / m) + ((lambda / m) * theta(j));
	end;
end;


% =============================================================

end
