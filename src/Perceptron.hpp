#pragma once

#include <vector>
#include <string>
#include <functional>


class Perceptron
{
public:
	typedef std::function<int(double)> ActivationFunction;

	Perceptron(unsigned int);

	void setActivationFunction(const ActivationFunction&);
	
	int process(const std::vector<double>&);

	void train(std::vector<std::vector<double>>, std::vector<int>, double, unsigned int);

	std::string toString();

private:
	const double INIT_MAX = 0.01;
	const double INIT_MIN = -0.01;

	void update(const std::vector<double>&, int, int, double);

	std::vector<double> weights;
	double bias;

	ActivationFunction activationFunction;


};
