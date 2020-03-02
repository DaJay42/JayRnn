package ch.dajay42.rnn;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.*;

import ch.dajay42.math.Util;
import ch.dajay42.math.function.DoubleTernaryOperator;
import ch.dajay42.math.linAlg.ColumnVectorDense;
import ch.dajay42.math.linAlg.Matrix;

@SuppressWarnings("WeakerAccess")
public class MinimalRnn implements Rnn, Serializable {

	private static final long serialVersionUID = 1L;
	
	private Matrix h; //hidden state
	
	//model parameters
	final Matrix Whh; //hidden to hidden
	final Matrix Wxh; //input to hidden
	final Matrix Why; //hidden to output

	final Matrix bh; // hidden bias
	final Matrix by; // output bias
	
	
	final Matrix mWxh;
	final Matrix mWhh;
	final Matrix mWhy;
	final Matrix mbh;
	final Matrix mby; //memory variables for Adagrad
	
	//hyperparameters
	final int h_size; // hidden size
	final int xy_size; // vocab size
	double learning_rate = 0.1d; //1e-1;
	static final double gradient_limit = 5.0d;
	private double dropout = 0.5d;
	private double smooth_loss;
	private double last_loss = Double.POSITIVE_INFINITY;
	long learnedSteps = 0L;
	
	/**inverse of prediction Temperature*/
	double beta = 1.0d;
	
	public long getLearnedSteps() {
		return learnedSteps;
	}
	
	// Adagrad update
	final static DoubleBinaryOperator Adagrad1 = 
			(mem, dparam) ->  mem + dparam * dparam;
	
	final DoubleTernaryOperator Adagrad2 =
			(param, dparam, mem) -> param - getLearningRate() * dparam / Math.sqrt(mem + 1e-8);
	
	// gradient clipping
	final static DoubleUnaryOperator clip =
			(a) -> Util.clamp(-gradient_limit, gradient_limit, a);
	
	
	public MinimalRnn(int hiddensize, int paramsize) {
		h_size = hiddensize;
		xy_size = paramsize;
		h = new ColumnVectorDense(h_size);
		Whh = Matrix.random(h_size, h_size, -0.01, 0.01);
		Wxh = Matrix.random(h_size, xy_size, -0.01, 0.01);
		Why = Matrix.random(xy_size, h_size, -0.01, 0.01);
		
		bh = new ColumnVectorDense(h_size);
		by = new ColumnVectorDense(xy_size);

		mWxh = Matrix.zeroesLike(Wxh);
		mWhh = Matrix.zeroesLike(Whh);
		mWhy = Matrix.zeroesLike(Why);
		mbh = Matrix.zeroesLike(bh); 
		mby = Matrix.zeroesLike(by);
		
		smooth_loss = -Math.log(1.0/xy_size);
	}

	@Override
	public Matrix step(Matrix x) {
		//update the hidden state
		//h = tanh(Wxh*x + Whh*h + bh)
	    h = Wxh.multiplySimple(x).inplaceSum(Whh.multiplySimple(h)).inplaceSum(bh);
	    h.inplaceElementWise(Math::tanh);
	    //compute the output vector
	    //y = Why*h + by
		return Why.multiplySimple(h).inplaceSum(by);
	}

	@Override
	public Matrix[] learn(Matrix[] in, Matrix[] expectedOut, Matrix h_in) {
		if(in.length != expectedOut.length){
			throw new IllegalArgumentException("Array dimensions must agree!");
		}
		
		//TODO: learning rate
		
		
		//loss function
		int inputs = in.length;
		
		int[] expectedIndex = new int[inputs];
		for(int t = 0; t < inputs; t++)
			expectedIndex[t] = RnnEncDec.indexOf(expectedOut[t]);
		
		// drop-out matrices
		boolean doDropout = dropout > 0;
		double p = 1-dropout; //chance of being 1
		Matrix pWhy = Why;
		
		if(doDropout){
			pWhy = Why.elementWise(Util::multiplication, Matrix.bernoulliLike(Why, p));
		}
		
		Matrix[] xs = new Matrix[inputs],
				 hs = new Matrix[inputs],
				 ys = new Matrix[inputs],
				 ps = new Matrix[inputs];
		
		h = h_in;
		hs[0] = h;
		double loss = 0;
		
		
		// forward pass
		for(int t = 1; t < inputs; t++){
			xs[t] = in[t-1];
			
			//hs[t] = tanh(Wxh*xs[t] + Whh*hs[t-1] + bh)
		    hs[t] = Whh.multiplySimple(hs[t-1]).inplaceSum(Wxh.multiplySimple(xs[t])).inplaceSum(bh);
		    hs[t].inplaceElementWise(Math::tanh);
		    
		    //ys[t] = Why*hs[t] + by
		    ys[t] = pWhy.multiplySimple(hs[t]);
		    if(doDropout) ys[t].scalarOp(Util::division, p);
		    ys[t].inplaceSum(by); // unnormalized log probabilities for next chars
		    
		    //ps[t] = exp(ys[t]) / sum(exp(ys[t]))
		    Matrix expYsT = ys[t].elementWise(Math::exp);
		    ps[t] = expYsT.scalarOp(Util::division, expYsT.aggregateOp(Util::sum)); // probabilities for next chars
		    
		    // loss += -log(ps[t].value[indexOf(expectedOut[t])][0])
		    loss += -Math.log(ps[t].getValueAt(expectedIndex[t])); // softmax (cross-entropy loss)
		}

		// backward pass: compute gradients going backwards
		Matrix dWxh = Matrix.zeroesLike(Wxh), 
				dWhh = Matrix.zeroesLike(Whh), 
				dWhy = Matrix.zeroesLike(Why),
				dbh = Matrix.zeroesLike(bh),
				dby = Matrix.zeroesLike(by),
				dhnext = Matrix.zeroesLike(h),
				dy,
				dh,
				dhraw;
		
		for(int t = inputs-1; t > 0; t--){ //reverse iteration
			//dy = copyOf(ps[t])
			dy = ps[t].clone();
			
			dy.modValueAt(expectedIndex[t],  -1.0); // backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
			
			//dWhy += dy*(hs[t]^T)
			dWhy.inplaceSum(dy.multiplySimple(hs[t].transpose()));
			
			//dby += dy
			dby.inplaceSum(dy);
			
			//dh = (Why^T)*dy + dhnext
			dh = (pWhy.transpose().multiplySimple(dy)).inplaceSum(dhnext); // backprop into h
			
			//dhraw = (1 - hs[t].^2) .* dh //mind the operator precedence
			dhraw = (hs[t].scalarOp(Math::pow, 2)).inplaceElementWise(a -> 1 - a).inplaceElementWise(Util::multiplication, dh); // backprop through tanh nonlinearity
			
			//dbh += dhraw
			dbh.inplaceSum(dhraw);
			
			//dWxh += dhraw*(xs[t]^T)
			dWxh.inplaceSum(dhraw.multiplySimple(xs[t].transpose()));
			
			//dWhh += dhraw*(hs[t-1]^T)
			dWhh.inplaceSum(dhraw.multiplySimple(hs[t-1].transpose()));
			
			//dhnext = (Whh^T)*dhraw
			dhnext = Whh.transpose().multiplySimple(dhraw);
		}
		 
		// clip to mitigate exploding gradients
		dWxh.inplaceElementWise(clip); 
		dWhh.inplaceElementWise(clip);
		dWhy.inplaceElementWise(clip); 
		dbh.inplaceElementWise(clip);
		dby.inplaceElementWise(clip);
		
		//update loss + learning rate
		loss /= inputs;
		//*
		if(loss > last_loss)
			learning_rate *= 0.9998;
		else
			learning_rate *= 1.0001;
		if(learning_rate < 1e-32)
			learning_rate = 1e-32;
		//*/
		last_loss = loss;
		smooth_loss = smooth_loss * 0.9 + loss * 0.1;
		
		
		// perform parameter update with Adagrad
		mWxh.inplaceElementWise(Adagrad1, dWxh);
		Wxh.inplaceElementWise(Adagrad2, dWxh, mWxh);
		
		mWhh.inplaceElementWise(Adagrad1, dWhh);
		Whh.inplaceElementWise(Adagrad2, dWhh, mWhh);
		
		mWhy.inplaceElementWise(Adagrad1, dWhy);
		Why.inplaceElementWise(Adagrad2, dWhy, mWhy);
		
		mbh.inplaceElementWise(Adagrad1, dbh);
		bh.inplaceElementWise(Adagrad2, dbh, mbh);
		
		mby.inplaceElementWise(Adagrad1, dby);
		by.inplaceElementWise(Adagrad2, dby, mby);
		//
		
		learnedSteps += inputs;
		//return predicted values for live sampling
		return ys;
	}

	@Override
	public List<Matrix> sample(Matrix h, Matrix[] seed, int n) {
		this.h = h;
		ArrayList<Matrix> ret = new ArrayList<>();
		for(int i = 0; i < seed.length-1; i++){
			step(seed[i]);
			ret.add(seed[i]);
		}

		Matrix r = seed[seed.length-1];
		ret.add(r);
		
		for(int i = 0; i < n; i++){
			r = step(r);
			r = softmax(r);
			ret.add(r);
		}
		return ret;
	}
	
	Matrix softmax(Matrix vector){
		//x = beta * vector //temperature-adjusted probabilities
		//v = exp(x) / sum(exp(x))
		Matrix p = vector.scalarOp(Util::multiplication, beta).inplaceElementWise(Math::exp); //beware of overflows
		double scalar = p.aggregateOp(Util::sum);
		double[] v = p.scalarOp(Util::division, scalar).getValuesInColumn(0);
		int index = 0;
		double selection = ThreadLocalRandom.current().nextDouble();
		for(int i = 0; i < v.length; i++){
			selection -= v[i];
			if(selection < 0){
				index = i;
				break;
			}
		}
		Matrix x = Matrix.zeroesLike(vector, true);
		x.setValueAt(index, 1.0);
		return x;
	}

	@Override
	public int getHiddenSize() {
		return h_size;
	}

	@Override
	public double getLastLoss() {
		return smooth_loss;
	}

	@Override
	public Matrix getH() {
		return h;
	}

	@Override
	public void setH(Matrix h) {
		this.h = h;
	}

	@Override
	public double getLearningRate() {
		return learning_rate;
	}

	@Override
	public void setLearningRate(double learningRate) {
		learning_rate = learningRate;
	}

	/**Set the prediction Temperature
	 * @param t Temperature, positive.
	 * */
	public void setTemperature(double t){
		if(t > 0)
			beta = 1/t;
		else
			throw new IllegalArgumentException("Argument must be between positive");
	}
	
	public double getTemperature(){
		return 1/beta;
	}

	public double getDropout() {
		return dropout;
	}

	public void setDropout(double dropout) {
		this.dropout = Util.clamp(0d, 1d, dropout);
	}
}
