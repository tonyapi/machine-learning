function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X_add = [ones(m,1),X];
z2 = X_add * Theta1';
a2 = sigmoid(z2);
a2_add = [ones(m,1),a2];
z3 = a2_add * Theta2';
a3 = sigmoid(z3);

y_new = zeros(m,num_labels);

for i = 1:num_labels
	y_new(:,i) = y == i;
end;

sum1 = sum(sum(-y_new .* log (a3)));
sum2 = sum(sum((1-y_new) .* log (1-a3)));

temp1 = Theta1(:,2:end);
temp2 = Theta2(:,2:end);

sum4 = sum(sum(temp1 .^ 2));
sum5 = sum(sum(temp2 .^ 2));

J = ((1/m) * (sum1 - sum2)) + ((lambda / (2 * m)) * (sum4 + sum5));

Del1 = 0;
Del2 = 0;

for i = 1:m
	a1new = X(i,:); 
	a1new_add = [1,a1new]; 
	z2new = a1new_add * Theta1'; 
	a2new = sigmoid(z2new); 
	a2new_add = [1,a2new]; 
	z3new = a2new_add * Theta2'; 
	a3new = sigmoid(z3new); 
%	delta3 = zeros(m,num_labels); % not necessary
	delta3(i,:) = a3new - y_new(i,:); 
	Theta2_new = Theta2(:,2:end); 
	delta2(i,:) = (delta3(i,:) * Theta2_new) .* (a2new .* (1 - a2new)); 
	Del2 = Del2 + delta3(i,:)' * a2new_add;
	Del1 = Del1 + delta2(i,:)' * a1new_add;
end;

Theta1_grad = (Del1 ./ m) + (Theta1 .* (lambda/m)); %this is the gradient for theta1
Theta1_grad(:,1) = Del1(:,1) ./ m; %this is the gradient for theta1, bias term
Theta2_grad = (Del2 ./ m) + (Theta2 .* (lambda/m)); %this is the gradient for theta2
Theta2_grad(:,1) = Del2(:,1) ./ m; %this is the gradient for theta2, bias term

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
