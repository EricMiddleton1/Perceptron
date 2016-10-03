#include <iostream>
#include <vector>

#include "Perceptron.hpp"

using namespace std;

int main() {
	vector<vector<double>> trainingInput = {	{1.5, 2.0},
																						{2.0, 3.5},
																						{3.0, 5.0},
																						{3.5, 2.5},
																						{4.5, 5.0},
																						{5.0, 7.0},
																						{5.5, 8.0},
																						{6.0, 6.0},
																						{4.0, 6.0}};
	vector<int> trainingOutput = {	-1,
																	-1,
																	-1,
																	-1,
																	1,
																	1,
																	1,
																	1,
																	1};
	
	Perceptron percept(2);

	percept.setActivationFunction([](double input) {
		if(input >= 0)
			return 1;
		else
			return -1;
	});

	double alpha = 0.001;
	unsigned int maxEpoch = 1000;

	cout << "Initial Perceptron state:\n" << percept.toString() << "\n\n";

	cout << "[Info] Training Perceptron with training set of size " << trainingInput.size() << endl;

	percept.train(trainingInput, trainingOutput, alpha, maxEpoch);

	cout << "[Info] Training Complete\n\nNew Perceptron state:\n" << percept.toString() << endl;
	
	cout << "\nEnter space-separated input vectors:" << endl;

	while(true) {
		cout << '>';

		double age, income;

		cin >> age;
		cin >> income;

		int result = percept.process({age, income});

		cout << '\t' << result << endl;
	}


	return 0;
}
