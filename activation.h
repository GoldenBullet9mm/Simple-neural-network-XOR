#pragma once
#include <vector>
using std::vector;

vector<vector<double>> sigmoid(vector<vector<double>> &input);
vector<vector<double>> sigmoid_d(vector<vector<double>> &input);
vector<vector<double>> relu(vector<vector<double>> &input);
vector<vector<double>> reluPrime(vector<vector<double>>&input);