#include "activation.h"

vector<vector<double>> sigmoid(vector<vector<double>> &input) {

	vector<vector<double>> output = vector<vector<double>>(input.size(), vector<double>(input[0].size()));

	for (int i = 0; i < input.size(); i++) {

		for (int j = 0; j < input[i].size(); j++) {

			output[i][j] = 1 / (1 + exp(-input[i][j]));
		}
	}

	return output;
}

vector<vector<double>> sigmoid_d(vector<vector<double>> &input) {

	vector<vector<double>> output = vector<vector<double>>(input.size(), vector<double>(input[0].size()));

	for (int i = 0; i < input.size(); i++) {

		for (int j = 0; j < input[i].size(); j++) {

			output[i][j] = input[i][j] * (1 - input[i][j]);
		}
	}

	return output;
}

vector<vector<double>> relu(vector<vector<double>>& input) {

	vector<vector<double>> output = vector<vector<double>>(input.size(), vector<double>(input[0].size()));

	for (int i = 0; i < input.size(); i++) {

		for (int j = 0; j < input[i].size(); j++) {

			if (input[i][j] < 0) {

				output[i][j]= 0.0;
			}

			else output[i][j]= input[i][j];
		}
	}
	return output;
}

vector<vector<double>> reluPrime(vector<vector<double>>&input) {

	vector<vector<double>> output = vector<vector<double>>(input.size(), vector<double>(input[0].size()));

	for (int i = 0; i < input.size(); i++) {

		for (int j = 0; j < input[i].size(); j++) {

			if (input[i][j] <= 0) {

				output[i][j]= 0.0;
			}

			else output[i][j]= 1.0;
		}
	}
	return output;
}