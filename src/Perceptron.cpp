#include "Perceptron.hpp"

#include <algorithm>
#include <iostream>
#include <cmath>

using namespace std;

Perceptron::Perceptron(unsigned int _inputCount)
	:	activationFunction{[](double input){
			return (input >= 0.) ? 1 : -1;
	}} {
	
	for(unsigned int i = 0; i < _inputCount; ++i) {
		double rnd = (rand() % 10001) / 10000. * (INIT_MAX - INIT_MIN) + INIT_MIN;
		weights.push_back(rnd);
	}

	bias = (rand() % 10001) / 10000. * (INIT_MAX - INIT_MIN) + INIT_MIN;
}

void Perceptron::setActivationFunction(const Perceptron::ActivationFunction& _activationFunction) {
	activationFunction = _activationFunction;
}

int Perceptron::process(const std::vector<double>& _input) {
	if(_input.size() != weights.size()) {
		//TODO: throw exception

		return 0;
	}

	double sum = 0.;

	for(unsigned int i = 0; i < weights.size(); ++i) {
		sum += weights[i] * _input[i];
	}

	sum += bias;

	return activationFunction(sum);
}

void Perceptron::train(vector<vector<double>> _inputs, vector<int> _outputs, double _alpha,
	unsigned int _maxEpoch) {
	vector<unsigned int> indicies;

	if(_inputs.size() < _outputs.size() || _inputs[0].size() != weights.size()) {
		cout << "[Error] Perceptron::train: input and output vector must be the same size!" << endl;
		return;
	}

	for(unsigned int i = 0; i < _outputs.size(); ++i) {
		indicies.push_back(i);
	}

	for(unsigned int epoch = 0; epoch < _maxEpoch; ++epoch) {
		//Shuffle the indicies
		random_shuffle(indicies.begin(), indicies.end());
		
		for(auto& i : indicies) {
			//Get the computed and desired outputs
			int computed = process(_inputs[i]);
			int desired = _outputs[i];

			//Update the perceptron
			update(_inputs[i], computed, desired, _alpha);
		}
	}
}

void Perceptron::update(const vector<double>& _input, int _computed, int _desired, double _alpha) {
	if(_computed == _desired) {
		//Nothing to update
		return;
	}

	int delta = _computed - _desired;
	double correction = _alpha * delta;

	//Update weights
	for(unsigned int i = 0; i < weights.size(); ++i) {
		if(_computed > _desired && _input[i] >= 0.)
			weights[i] -= correction * _input[i];
		else if(_computed > _desired && _input[i] < 0.)
			weights[i] += correction * _input[i];
		else if(_computed < _desired && _input[i] >= 0)
			weights[i] -= correction * _input[i];
		else
			weights[i] += correction * _input[i];
	}

	//Update bias
	bias -= correction;
}

string Perceptron::toString() {
	string str;

	str = "Input Weights:\n";

	for(unsigned int i = 0; i < weights.size(); ++i) {
		str += "[" + to_string(i) + "]\t\t" + to_string(weights[i]) + "\n";
	}

	str += "Bias:\t\t" + to_string(bias);

	return str;
}
