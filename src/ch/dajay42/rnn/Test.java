package ch.dajay42.rnn;

import java.util.List;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;

import ch.dajay42.collections.RandomizedTreeSet;
import ch.dajay42.math.Matrix;
import ch.dajay42.math.MatrixNxM;
import ch.dajay42.rnn.MinimalRnn;
import ch.dajay42.rnn.RnnEncDec;

public class Test {

	public static void main(String[] args) {
		inspectChars();
	}
	
	public static void inspectChars(){
		char[] chars = new char[256];
		for(short i = 0; i < 256; i++){
			chars[i] = (char) i;
		}
		for(byte i = 0; i < 16; i++){
			System.out.print("\n");
			System.out.print(i*16+":\t");
			for(byte j = 0; j < 16; j++){
				System.out.print(chars[i*16+j]);
			}
		}
		System.out.println("");
	}
	
	public static void testMinRnn(){
		int h = 64;
		int n = 256;
		
		List<String> lines = null;
		try {
			lines = Files.readAllLines(FileSystems.getDefault().getPath("res/shakespear.txt"));
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		String text = String.join("\n", lines);
		
		RandomizedTreeSet<Character> charset = new RandomizedTreeSet<Character>();
		for(char c : text.toCharArray()){
			charset.add(c);
		}
		//for(String s : charset.prettyPrint()){
		//	System.out.println(s);
		//}
		
		RnnEncDec<Character> encdec = new RnnEncDec<Character>(charset);
		MinimalRnn rnn = new MinimalRnn(h, charset.size());
		
		int k = 1000;//chunk size
		int kk = text.length()-1;
		int offset = 0;
		
		Matrix[] textEncd = new Matrix[kk];
		for(int j = 0; j < kk; j++){
			textEncd[j] = encdec.encode(text.charAt(j));
		}
		
		Matrix[] in = new Matrix[k];
		Matrix[] exout = new Matrix[k];

		/*Matrix seed = encdec.encode(text.charAt(0));
		
		for(int i = 0; i < k; i++){
			in[i] = encdec.encode(text.charAt(i));
			exout[i] = encdec.encode(text.charAt(i));
		}*/
		
		/*List<Matrix> out1 = rnn.sample(new Matrix(h,1), seed, n);
		
		for(Matrix m : out1){
			Character c = encdec.decode(m);
			System.out.append(m.toString());
		}*/

		System.out.append('\n');
		if(true){
		
		for(int i = 0; i < Integer.MAX_VALUE; i++){
			Matrix seed = encdec.encode(text.charAt(offset % kk));
			
			for(int j = 0; j < k; j++){
				int jk = (j+offset)%kk;
				in[j] = textEncd[jk];
				exout[j] = textEncd[jk];
			}
			offset += k;
			
			Matrix hmat = new MatrixNxM(h,1);
			
			rnn.learn(in, exout, hmat);
			
			if(i % 10 == 0){
				System.out.append('\n');
				System.out.append("i = "+i);
				System.out.append('\n');
				//List<MatrixNxM> out = rnn.sample(new MatrixNxM(h,1), seed, n);
				
				rnn.setH(hmat);
				Matrix r = seed;
				for(int j = 0; j < n; j++){
					r = rnn.step(r);
					Character c = encdec.decode(r);
					System.out.append(c);
					r = encdec.encode(c);
				}
				
				/*for(MatrixNxM m : out){
					Character c = encdec.decode(m);
					System.out.append(c);
				}*/

				System.out.append('\n');
				System.out.append('\n');
			}
			System.out.append("Loss: "+ rnn.getLastLoss());
			System.out.append('\n');
			System.out.flush();
		}}
	}
}
