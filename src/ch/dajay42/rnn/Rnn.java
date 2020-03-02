package ch.dajay42.rnn;

import java.util.List;

import ch.dajay42.math.linAlg.Matrix;

public interface Rnn{

	Matrix step(Matrix x);

	Matrix[] learn(Matrix[] in, Matrix[] exout, Matrix h);

	List<Matrix> sample(Matrix h, Matrix seed[], int n);

	int getHiddenSize();
	
	double getLastLoss();
	
	long getLearnedSteps();
	
	double getLearningRate();
	
	void setLearningRate(double learningRate);
	
	void setTemperature(double t);
	
	double getTemperature();

	Matrix getH();

	void setH(Matrix h);
}
