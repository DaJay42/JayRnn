package ch.dajay42.rnn;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import ch.dajay42.math.Util;
import ch.dajay42.math.linAlg.ColumnVectorSparse;
import ch.dajay42.math.linAlg.Matrix;

public class RnnEncDec<E extends Comparable<E>> {

	private final Map<E, Integer> forward = new HashMap<>();
	private final Map<Integer, E> backward = new HashMap<>();
	
	final int classes;
	
	/**inverse of prediction Temperature*/
	private double beta = 1;
	
	@SuppressWarnings("unchecked")
	RnnEncDec(Set<E> elements) {
		Object[] elems = elements.toArray();
		Arrays.sort(elems);
		classes = elems.length;
		
		for(Integer i = 0; i < classes; i++){
			forward.put((E) elems[i], i);
			backward.put(i, (E) elems[i]);
		}
		
	}
	
	Matrix encode(E item){
		Matrix v = new ColumnVectorSparse(classes);
		int e = forward.get(item);
		v.setValueAt(e, 1);
		return v;
	}

	E decode(Matrix vector){
		//x = beta * vector //temperature-adjusted probabilities
		//v = exp(x) / sum(exp(x))
		Matrix p = vector.scalarOp(Util::multiplication, beta).inplaceElementWise(Math::exp); //beware of overflows
		double scalar = p.aggregateOp(Util::sum);
		double[] v = p.scalarOp(Util::division, scalar).getValuesInColumn(0);
		
		double selection = ThreadLocalRandom.current().nextDouble();
		for(int i = 0; i < v.length; i++){
			selection -= v[i];
			if(selection < 0)
				return backward.get(i);
		}
		return null;
	}

	E decodeMax(Matrix r){
		double max = Double.MIN_VALUE;
		int j = 0;
		for(int i = 0; i < r.rows; i++){
			if(r.getValueAt(i,0) > max){
				max = r.getValueAt(i,0);
				j = i;
			}
		}
		return backward.get(j);
	}
	
	static int indexOf(Matrix vector){
		for(int i = 0; i < vector.rows; i++){
			if(vector.getValueAt(i,0) >= 1){
				return i;
			}
		}
		return -1;
	}

	/**Set the prediction Temperature
	 * @param t Temperature, positive.
	 * */
	public void setTemperature(double t){
		if(t <= 0) throw new IllegalArgumentException("Argument must be positive");
		beta = 1/t;
	}
}
