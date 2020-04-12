function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

temp = 0;
for i=1:m
	temp = temp + (-y(i) * log (1 ./ (1 + (e .^ (- X(i,:) * theta)))) ...
	- (1 - y(i)) * log (1 - (1 ./ (1 + (e .^ (- X(i,:) * theta))))));
	end;
J = temp / m;

i=0;
thetaone = 0;
for i=1:m
	thetaone = thetaone + (1 ./ (1 + (e .^ (- X(i,:) * theta))) - y(i));
	end;
grad(1) = thetaone / m;

i=0;
thetatwo = 0;
for i=1:m
	thetatwo = thetatwo + ((1 ./ (1 + (e .^ (- X(i,:) * theta))) - y(i)) * X(i,2));
	end;
grad(2) = thetatwo / m;

i=0;
thetathree = 0;
for i=1:m
	thetathree = thetathree + ((1 ./ (1 + (e .^ (- X(i,:) * theta))) ...
	- y(i)) * X(i,3));
	end;
grad(3) = thetathree / m;

% =============================================================

end
