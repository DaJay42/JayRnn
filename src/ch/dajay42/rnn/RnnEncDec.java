package ch.dajay42.rnn;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import ch.dajay42.collections.RandomizedTreeMap;
import ch.dajay42.math.Util;
import ch.dajay42.math.VectorNSparse;
import ch.dajay42.math.Matrix;

@SuppressWarnings("unused")
public class RnnEncDec<E extends Comparable<E>> {

	Map<E, Integer> forward = new HashMap<E, Integer>();
	Map<Integer, E> backward = new HashMap<Integer, E>();
	
	int n;
	
	/**inverse of prediction Temperature*/
	double beta = 1;
	
	@SuppressWarnings("unchecked")
	public RnnEncDec(Set<E> elements) {
		Object[] elems = elements.toArray();
		Arrays.sort(elems);
		n = elems.length;
		
		for(Integer i = 0; i < n; i++){
			forward.put((E) elems[i], i);
			backward.put(i, (E) elems[i]);
		}
		
	}
	
	public Matrix encode(E item){
		Matrix e = new VectorNSparse(n);
		int i = forward.get(item);
		e.setValueAt(i,0, 1);
		return e;
	}

	public E decode(Matrix vector){
		//x = beta * vector //temperature-adjusted probabilities
		//v = exp(x) / sum(exp(x))
		Matrix p = vector.scalarOp(Util.MULTIPLICATION, beta).inplaceElementWise(Math::exp); //beware of overflows
		double scalar = p.aggregateOp(Util::sum);
		double[] v = p.scalarOp(Util.DIVISION, scalar).getValuesInColumn(0);
		
		double selection = ThreadLocalRandom.current().nextDouble();
		for(int i = 0; i < v.length; i++){
			selection -= v[i];
			if(selection < 0)
				return backward.get(i);
		}
		return null;
	}

	public E decodeMax(Matrix r){
		double max = Double.MIN_VALUE;
		int j = 0;
		for(int i = 0; i < r.n; i++){
			if(r.getValueAt(i,0) > max){
				max = r.getValueAt(i,0);
				j = i;
			}
		}
		return backward.get(j);
	}
	
	public static int indexOf(Matrix vector){
		for(int i = 0; i < vector.n; i++){
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
		if(t <= 0) throw new IllegalArgumentException("Argument must be between positive");
		beta = 1/t;
	}
}
