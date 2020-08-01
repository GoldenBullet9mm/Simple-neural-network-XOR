#include "vector_functions.h"

double random_number(double low, double high) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(low, high);
	return dis(gen);
}

vector<vector<double>> random_weights(int n_neurons, int n_inputs) {

	vector<vector<double>>weights = vector<vector<double>>(n_inputs, vector<double>(n_neurons));

	for (int i = 0; i < n_inputs; i++) {

		for (int j = 0; j < n_neurons; j++) {
			
			weights[i][j] = 0.1 * random_number(-1, 1);
		}
	}
	return weights;
}

vector<vector<double>> dot(vector<vector<double>> inputs, vector<vector<double>> weights) {

	vector<vector<double>> outputs = vector<vector<double>>(inputs.size(), vector<double>(weights[0].size()));

	for (int i = 0; i < inputs.size(); i++) {

		for (int j = 0; j < weights[0].size(); j++) {

			double temp = 0;

			for (int k = 0; k < inputs[0].size(); k++) {

				temp += inputs[i][k] * weights[k][j];
			}

			outputs[i][j] = temp;
		}
	}
	return outputs;
}

vector<vector<double>> transpose(vector<vector<double>> input) {

	vector<vector<double>> output = vector<vector<double>>(input[0].size(), vector<double>(input.size()));

	for (int i = 0; i < input.size(); i++) {

		for (int j = 0; j < input[i].size(); j++) {

			output[j][i] = input[i][j];
		}
	}
	return output;
}

vector<vector<double>> operator + (vector<vector<double>> vector1, vector<double> vector2) {

	for (int i = 0; i < vector1.size(); i++) {

		vector1[i] = vector1[i] + vector2;
	}

	return vector1;
}

vector<double> operator + (vector<double> vector1, vector<double> vector2) {

	vector<double> output(vector1.size());

	for (int i = 0; i < vector1.size(); i++) {

		output[i] = vector1[i] + vector2[i];
	}

	return output;
}

vector<vector<double>> operator - (vector<vector<double>> vector1, vector<vector<double>> vector2) {

	vector<vector<double>> output = vector<vector<double>>(vector1.size(), vector<double>(vector1[0].size()));

	for (int i = 0; i < vector1.size(); i++) {

		for (int j = 0; j < vector1[i].size(); j++) {

			output[i][j] = vector1[i][j] - vector2[i][j];
		}
	}
	return output;
}

vector<double> operator - (vector<double> vector1, vector<double> vector2) {

	vector<double> output(vector1.size());

	for (int i = 0; i < vector1.size(); i++) {

		output[i] = vector1[i] - vector2[i];
	}

	return output;
}

vector<vector<double>> operator * (vector<vector<double>> vector1, vector<vector<double>> vector2) {

	vector<vector<double>> output = vector<vector<double>>(vector1.size(), vector<double>(vector1[0].size()));

	for (int i = 0; i < vector1.size(); i++) {

		for (int j = 0; j < vector1[i].size(); j++) {

			output[i][j] = vector1[i][j] * vector2[i][j];
		}
	}
	return output;
}

vector<vector<double>> operator * (double num, vector<vector<double>> vector2) {

	vector<vector<double>> output = vector<vector<double>>(vector2.size(), vector<double>(vector2[0].size()));

	for (int i = 0; i < vector2.size(); i++) {

		for (int j = 0; j < vector2[i].size(); j++) {

			output[i][j] =  num * vector2[i][j];
		}
	}
	return output;
}

vector<double> operator * (double num, vector<double> vector2) {

	vector<double> output(vector2.size());

	for (int i = 0; i < vector2.size(); i++) {

		output[i] = num * vector2[i];
	}

	return output;
}

vector<double> sum(vector<vector<double>> vector1) {

	vector<double>output(vector1[0].size());

	for (int i = 0; i < vector1.size(); i++) {

		double temp = 0;

		for (int j = 0; j < vector1[i].size(); j++) {

			temp = vector1[i][j];

			output[j] += temp;
		}
	}

	return output;
}

void print(vector<vector<double>> print_inputs) {

	cout << "------------------" << endl;

	for (int i = 0; i < print_inputs.size(); i++) {

		for (int j = 0; j < print_inputs[i].size(); j++) {

			cout << print_inputs[i][j] << " ";
		}
		cout << endl;
	}

	cout << "------------------" << endl;
}

double mse_loss (vector<vector<double>> vector1, int epoch) {

	double loss = 0;
	int cnt = 0;
	
	for (int i = 0; i < vector1.size(); i++) {

		for (int j = 0; j < vector1[i].size(); j++) {

			loss += vector1[i][j] * vector1[i][j];
			cnt ++;
		}
	}
	
	loss /= cnt;

	if (epoch % 100 == 0) {
		cout << "Epoch: " << epoch << ", loss: " << loss << endl;
		cout << "------------------------------" << endl;
	}
	
	return loss;
}

void print_prediction(vector<vector<double>>inputs, vector<vector<double>> expected_result) {

	for (int i = 0; i < inputs.size(); i++) {

		for (int j = 0; j < inputs[i].size(); j++) {

			cout << "Prediction: " << inputs[i][j] << ", expected result: " << expected_result[i][j];
		}

		cout << endl;
	}

}
