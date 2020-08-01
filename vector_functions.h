#pragma once
#include <random>
#include <vector>
#include <iostream>

using std::random_device;
using std::uniform_real_distribution;
using std::mt19937;
using std::vector;
using std::cout;
using std::endl;

vector<vector<double>> random_weights(int n_inputs, int n_neurons);
vector<vector<double>> dot(vector<vector<double>> weights, vector<vector<double>> inputs);
vector<vector<double>> transpose(vector<vector<double>> input);
vector<vector<double>> operator + (vector<vector<double>> vector1, vector<double> vector2);
vector<double> operator + (vector<double> vector1, vector<double> vector2);
vector<vector<double>> operator - (vector<vector<double>> vector1, vector<vector<double>> vector2);
vector<double> operator - (vector<double> vector1, vector<double> vector2);
vector<vector<double>> operator * (vector<vector<double>> vector1, vector<vector<double>> vector2);
vector<vector<double>> operator * (double num, vector<vector<double>> vector2);
vector<double> operator * (double num, vector<double> vector2);
vector<double> sum(vector<vector<double>> vector1);
void print(vector<vector<double>> print_inputs);
double mse_loss(vector<vector<double>> vector1, int epoch);
void print_prediction(vector<vector<double>> print_inputs, vector<vector<double>> expected_result);