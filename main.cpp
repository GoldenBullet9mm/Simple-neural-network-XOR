// Neural network XOR problem.
// Created by Dmitry Lebedev on 1/08/2020.
// Copyright © 2020 Dmitry Lebedev. All rights reserved.
#include <conio.h>
#include "activation.h"
#include "vector_functions.h"


int main() {

	vector<vector<double>> inputs{ { 0.0, 0.0, 0.0 },
	                               { 0.0, 1.0, 0.0 },
				       { 1.0, 0.0, 1.0 }, 
	                               { 1.0, 0.0, 0.0 },
	                               { 0.0, 1.0, 1.0 },
	                               { 1.0, 1.0, 1.0 } };
			
	vector<vector<double>> expected_result { { 0.0, 1.0, 1.0, 1.0, 1.0, 0.0 } };
	expected_result = transpose(expected_result);

	vector<vector<double>> hidden_layer;
	vector<vector<double>> hidden_weights = random_weights(16, 3); // The second layer has 16 neurons
	vector<double> hidden_bias =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	                                       
	vector<vector<double>> output_layer;
	vector<vector<double>> outputs_weights = random_weights(1, 16);// The third layer has one neuron
	vector<double> outputs_bias { 0 };
	

	int epoch = 501;
	double error_threshold = 0.0001;
	double learning_rate = 2.2;


	for (int i = 0; i < epoch; i++) {

		//Forward pass

		hidden_layer = relu(dot(inputs, hidden_weights) + hidden_bias);		

		output_layer = sigmoid(dot(hidden_layer, outputs_weights) + outputs_bias);

		//Backpropagation

		vector<vector<double>> error = output_layer - expected_result;

		vector<vector<double>> delta_output = error * sigmoid_d(output_layer);
		
		vector<vector<double>> delta_hidden_layer = dot(delta_output, transpose(outputs_weights)) * reluPrime(hidden_layer);

		//Update biases

		outputs_bias = outputs_bias - learning_rate * sum(delta_output);

		hidden_bias = hidden_bias - learning_rate * sum(delta_hidden_layer);

		//Update weights

		outputs_weights = outputs_weights - learning_rate * dot(transpose(hidden_layer), delta_output);

		hidden_weights = hidden_weights - learning_rate * dot(transpose(inputs), delta_hidden_layer);
	
		double loss = mse_loss(error, i);
		if (loss < error_threshold) {
			cout << "Epoch: " << i << " loss: " << loss << endl << endl;
			break;
		}
	}
	
	print_prediction(output_layer, expected_result);

	_getch();
	return 0;
}
