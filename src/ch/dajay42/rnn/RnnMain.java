package ch.dajay42.rnn;

import java.io.*;
import java.nio.file.*;
import java.util.*;

import ch.dajay42.collections.RandomizedTreeSet;
import ch.dajay42.math.*;

@SuppressWarnings("unused")
public class RnnMain {
	
	static RnnEncDec<Byte> asciiEncDec;

	static Rnn currentRnn = null;
	static String rnnLoadPath = "";
	static String rnnStorePath = "";
	
	static String textPath = "";
	static byte[] text = null;
	static int textLength = 0;
	static Matrix[] textEncd = null;
	static int offset = 0;
	
	static int seqLength = 32;
	static int autoSampleFrequency = 128;
	static int autoSampleSize = 256;
	static int blockSize = 1024*1024;
	
	final static String helpStr = "help";
	final static String quitStr = "quit";
	final static String statusStr = "status";
	
	final static String loadStr = "load";
	final static String storeStr = "store";
	final static String createStr = "create";
	
	final static String sampleStr = "sample";
	final static String learnStr = "learn";
	
	final static String readStr = "read";
	final static String writeStr = "write";
	
	final static String seqLengthStr = "setseqlength";
	final static String autoFreqStr = "setautosamplefreq";
	final static String autoSizeStr = "setautosamplesize";
	final static String blockSizeStr = "setblocksize";
	final static String temperatureStr = "settemperature";
	final static String learnRateStr = "setlearnrate";
	
	final static String settingsFile = "settings.parcel";

	
	static void initEncDec(){
		Set<Byte> charset = new HashSet<Byte>();
		for(byte s = 0; s >= 0; s++){
			charset.add(s);
		}
		asciiEncDec = new RnnEncDec<Byte>(charset);
	}
	
	static void loadTextBlock(){
		textEncd = new Matrix[blockSize];
		System.gc();
		System.out.println("Preparing text segment...");
		for(int j = 0; j < blockSize; j++){
			int k = (j + offset) % textLength;
			textEncd[j] = asciiEncDec.encode(text[k]);
			if(j % 65536 == 0)
				System.out.println(String.format("\r%.1f%%", (j*(100d/blockSize))));
		}
		System.out.println("Text segment prepared");
		/*for(int t = 0; t < blockSize; t++){
			System.out.print(asciiEncDec.decodeMax(textEncd[t]));
		}
		System.out.println();*/
	}
	
	public static void loadText(String filename){
		System.out.println("Loading text from \""+filename+"\"...");
		try {
			text = Files.readAllBytes(FileSystems.getDefault().getPath(filename));
		} catch (IOException e) {
			System.err.println(e);
			System.out.println("Error: Could not load text from file "+filename);
			return;
		}
		
		textLength = text.length;
		offset = 0;
		textPath = filename;
		textEncd = null;
		
		System.out.println("Text loaded.");
		loadTextBlock();
	}
	
	public static void loadSettings(String filename){
		SettingsParcel parcel = null;
		try {
			byte[] bytes = Files.readAllBytes(FileSystems.getDefault().getPath(filename));
			try(ObjectInputStream s = new ObjectInputStream(new ByteArrayInputStream(bytes))){
				parcel = (SettingsParcel) s.readObject();
			}
		} catch (IOException | ClassNotFoundException e) {
			System.out.println("No settings file found.");
			return;
		}
		seqLength = parcel.chunkSize;
		autoSampleFrequency = parcel.autoSampleFrequency;
		autoSampleSize = parcel.autoSampleSize;
		blockSize = parcel.blockSize;
	}
	
	public static void storeSettings(String filename){
		if(!filename.endsWith(".parcel"))
			filename = filename + ".parcel";
		
		SettingsParcel parcel = new SettingsParcel(seqLength, autoSampleFrequency, autoSampleSize,blockSize);
		try(ByteArrayOutputStream bs = new ByteArrayOutputStream()){
			try(ObjectOutputStream s = new ObjectOutputStream(bs)){
				s.writeObject(parcel);
				s.flush();
				byte[] bytes = bs.toByteArray();
				Files.write(FileSystems.getDefault().getPath(filename), bytes);
				
			}
		} catch (IOException e) {
			System.err.println(e);
			System.out.println("Error: Could not store Settings in file "+filename);
			return;
		}
	}
	
	public static void loadRnn(String filename){
		try {
			byte[] bytes = Files.readAllBytes(FileSystems.getDefault().getPath(filename));
			try(ObjectInputStream s = new ObjectInputStream(new ByteArrayInputStream(bytes))){
				Object o = s.readObject();
				currentRnn = (Rnn) o;
			}
		} catch (IOException | ClassNotFoundException e) {
			System.err.println(e);
			System.out.println("Error: Could not load RNN from file "+filename);
			return;
		}
		offset = 0;
		rnnLoadPath = filename;
		rnnStorePath = "";

		System.out.println("RNN loaded.");
	}
	
	public static void storeRnn(String filename){
		
		if(!filename.endsWith(".rnn"))
			filename = filename + ".rnn";
		
		try(ByteArrayOutputStream bs = new ByteArrayOutputStream()){
			try(ObjectOutputStream s = new ObjectOutputStream(bs)){
				s.writeObject(currentRnn);
				s.flush();
				byte[] bytes = bs.toByteArray();
				Files.write(FileSystems.getDefault().getPath(filename), bytes);
				
			}
		} catch (IOException e) {
			System.err.println(e);
			System.out.println("Error: Could not store RNN in file "+filename);
			return;
		}
		rnnStorePath = filename;
		System.out.println("RNN stored to '"+filename+"'.");
	}
	
	public static void createRnn(int hiddenSize){
		currentRnn = new MinimalRnn(hiddenSize, asciiEncDec.n);
		offset = 0;
		rnnLoadPath = "";
		rnnStorePath = "";
		System.out.println("Created new RNN.");
	}
	
	static void learn(int chunks){
		Matrix[] in = new Matrix[seqLength];
		Matrix[] exout = new Matrix[seqLength];
		

		System.out.append('\n');
		for(int i = 0; i < chunks; i++){
			
			if((offset % blockSize) + seqLength > blockSize){
				loadTextBlock();
				Matrix hmat = new VectorNSparse(currentRnn.getHiddenSize());
				currentRnn.setH(hmat);
			}
			
			Matrix last = null;
			for(int j = 0; j < seqLength; j++){
				int jk = (j + offset) % blockSize;
				in[j] = textEncd[jk];
				exout[j] = textEncd[jk];
				if(j == seqLength - 1)
					last = textEncd[jk];
			}
			offset += seqLength;
			offset %= textLength;
			
			
			currentRnn.learn(in, exout, currentRnn.getH());
			currentRnn.step(last);
			
			if(autoSampleFrequency > 0 && i % autoSampleFrequency == 0){
				Matrix[] seed = {asciiEncDec.encode(text[offset])};
				sample(autoSampleSize, seed);
			}
			System.out.append("Loss/step: "+ currentRnn.getLastLoss());
			
			System.out.append('\n');
			System.out.flush();
		}

	}
	
	static void write(String filename, int chars, Matrix[] seed){
		if(!filename.endsWith(".txt"))
			filename = filename + ".txt";
		
		try(ByteArrayOutputStream bs = new ByteArrayOutputStream()){
			try(PrintStream ps = new PrintStream(bs)){
				sample(chars, seed, ps);
				ps.flush();
				bs.flush();
				byte[] bytes = bs.toByteArray();
				Files.write(FileSystems.getDefault().getPath(filename), bytes);
			}
		} catch (IOException e) {
			System.err.println(e);
			System.out.println("Error: Could not write to file "+filename);
			return;
		}
		System.out.println("Wrote "+chars+" characters to '"+filename+"'.");
	}
	
	static void sample(int chars, Matrix[] seed){
		sample(chars, seed, System.out);
	}
	
	
	static void sample(int chars, Matrix[] seed, PrintStream out){
		
		out.append('\n');
		out.append("training steps = "+currentRnn.getLearnedSteps());
		out.append('\n');

		Matrix h = new VectorN(currentRnn.getHiddenSize());
		
		List<Matrix> res = currentRnn.sample(h, seed, chars);
		Byte b;
		for(Matrix r : res){
			b = asciiEncDec.decodeMax(r);
			out.write(b);
		}
		
		out.append('\n');
		out.append('\n');

	}
	
	static void status(){
		StringBuilder builder = new StringBuilder();
		builder.append("RnnMain:");
		builder.append('\n');
		builder.append('\n');
		builder.append("[Settings]");
		builder.append('\n');
		builder.append("chunkSize="+seqLength);
		builder.append('\n');
		builder.append("blockSize="+blockSize);
		builder.append('\n');
		builder.append("autoSampleFrequency="+autoSampleFrequency);
		builder.append('\n');
		builder.append("autoSampleSize="+autoSampleSize);
		builder.append('\n');
		builder.append('\n');
		builder.append("[RNN]");
		builder.append('\n');
		
		if(currentRnn != null){
			builder.append("size="+currentRnn.getHiddenSize());
			builder.append('\n');
			builder.append("steps="+currentRnn.getLearnedSteps());
			builder.append('\n');
			builder.append("current loss="+currentRnn.getLastLoss());
			builder.append('\n');
			builder.append("temperature="+currentRnn.getTemperature());
			builder.append('\n');
			builder.append("learningrate="+currentRnn.getLearningRate());
			builder.append('\n');
		}else{
			builder.append("null");
			builder.append('\n');
		}
		
		builder.append('\n');
		builder.append("[Text]");
		builder.append('\n');
		builder.append("path="+textPath);
		builder.append('\n');
		builder.append("length="+textLength);
		builder.append('\n');
		builder.append('\n');
		System.out.println(builder.toString());
	}
	
	static void help(){
		StringBuilder builder = new StringBuilder();
		builder.append("The following commands are available:\n");
		builder.append("\thelp\n");
		builder.append("\n");				
		builder.append("\tnew h\n");
		builder.append("\tstore filename\n");
		builder.append("\tload filename\n");
		builder.append("\n");
		builder.append("\tlearn n\n");
		builder.append("\tsample n [c]\n");
		builder.append("\n");
		builder.append("\t"+seqLengthStr+" n \n");
		builder.append("\t"+autoFreqStr+" n \n");
		builder.append("\t"+autoSizeStr+" n \n");
		builder.append("\n");
		builder.append("\tquit\n");
		builder.append("\n");
		builder.append("\n\n\n");
		
		builder.append("help: Displays this text.\n\n");
		builder.append("new: Discards the current RNN and creates an untrained RNN with <h> hidden internal states.\n\n");
		builder.append("store: Stores the current RNN in <filename>.\n\n");
		builder.append("load: Discardes the current RNN and loads the one stored in <filename>.\n\n");
		builder.append("learn: Learn from currently loaded text file for <n> chunks.\n");
		builder.append("sample: Samples and prints <n> characters from the RNN, starting from the seed character <c>, or newline.\n\n");
		builder.append("quit: Saves settings and terminates the program.\n\n");
		builder.append(seqLengthStr+": Sets size of chunks to <n> characters.\n\n");
		builder.append(autoFreqStr+": Sets 'learn' to auto-sample every <n> chunks.\n\n");
		builder.append(autoSizeStr+": Sets 'learn' to auto-sample for <n> characters.\n\n");
		System.out.println(builder.toString());
		
	}
	
	static boolean parseCommand(String[] words){
		try{
			switch(words[0].toLowerCase()){
			case(helpStr):
				help();
				break;
			case(statusStr):
				status();
				break;
			case(learnStr):
				if(currentRnn != null){
					if(text != null)
						learn(Integer.parseInt(words[1]));
					else
						System.out.println("Cannot learn: No text loaded.");
				}else{
					System.out.println("Cannot learn: No RNN loaded.");
				}
				break;
			case(loadStr):
				loadRnn(words[1]);
				break;
			case(createStr):
				createRnn(Integer.parseInt(words[1]));
				break;
			case(quitStr):
				System.out.println("Saving settings...");
				System.out.println("Quitting.");
				return false;
			case(sampleStr):
				if(currentRnn != null){
					String seedStr = (words.length > 2) ? words[2] : "\n";
					byte[] seeds = seedStr.getBytes();
					Matrix[] seed = new Matrix[seedStr.length()];
					for(int i = 0; i < seeds.length; i++){
						seed[i] = asciiEncDec.encode(seeds[i]);
					}
					sample(Integer.parseInt(words[1]), seed);
				}else{
					System.out.println("Cannot sample: No RNN loaded.");
				}
				break;
			case(storeStr):
				if(currentRnn != null){
					storeRnn(words[1]);
				}else{
					System.out.println("Cannot store RNN: No RNN loaded.");
				}
				break;
			case(seqLengthStr):
				seqLength = Integer.parseInt(words[1]);
				System.out.println("Set seqLength to "+seqLength+".");
				break;
			case(blockSizeStr):
				blockSize = Integer.parseInt(words[1]);
				System.out.println("Set blockSize to "+blockSize+".");
				break;
			case(autoFreqStr):
				autoSampleFrequency = Integer.parseInt(words[1]);
				System.out.println("Set autoSampleFrequency to "+autoSampleFrequency+".");
				break;
			case(autoSizeStr):
				autoSampleSize = Integer.parseInt(words[1]);
				System.out.println("Set autoSampleSize to "+autoSampleSize+".");
				break;
			case(temperatureStr):
				if(currentRnn != null){
					double t = Double.parseDouble(words[1]);
					currentRnn.setTemperature(t);
					System.out.println("Set temperature to "+t+".");
				}else
					System.out.println("Cannot set temperature: no RNN loaded.");
				break;
			case(learnRateStr):
				if(currentRnn != null){
					double r = Double.parseDouble(words[1]);
					currentRnn.setLearningRate(r);
					System.out.println("Set learning rate to "+r+".");
				}else
					System.out.println("Cannot set learning rate: no RNN loaded.");
				break;
			case(readStr):
				loadText(words[1]);
				break;
			case(writeStr):
				if(currentRnn != null){
					int chars = (words.length > 2) ? Integer.parseInt(words[2]) : autoSampleSize;
	
					String seedStr = (words.length > 3) ? seedStr = words[3] : "\n\n";
					byte[] seeds = seedStr.getBytes();
					Matrix[] seed = new Matrix[seedStr.length()];
					for(int i = 0; i < seeds.length; i++){
						seed[i] = asciiEncDec.encode(seeds[i]);
					}
					
					write(words[1], chars, seed);
				}else{
					System.out.println("Cannot write: No RNN loaded.");
				}
				break;
			default:
				System.err.println("Unknown command. Enter 'help' to get a list of commands.");
				System.out.println();
			}
		}catch(Exception e){
			e.printStackTrace(System.err);
			System.err.println("Failed to parse command. Enter 'help' to get a list of commands.");
			System.out.println();
		}
		return true;
	}
	
	public static void main(String[] args) {
		System.out.println("Loading...");
		
		initEncDec();
		System.out.println("...");
		
		loadSettings(settingsFile);
		System.out.println("...");
		
		boolean b = true;
		String s = "";

		System.out.println("RnnMain: Ready.");
		
		
		try(Scanner in = new Scanner(System.in)){
			while(b){
				System.err.flush();
				System.out.flush();
				System.out.print("?: ");
				s = in.nextLine();
				b = parseCommand(s.split(" "));
			}
		}
		storeSettings(settingsFile);
	}

	
	static class SettingsParcel implements Serializable{
		private static final long serialVersionUID = 1L;
		int chunkSize = 32;
		int autoSampleFrequency = 100;
		int autoSampleSize = 256;
		int blockSize = 256*1024;
		
		SettingsParcel(int chunkSize, int autoSampleFrequency, int autoSampleSize, int blockSize){
			this.chunkSize = chunkSize;
			this.autoSampleFrequency = autoSampleFrequency;
			this.autoSampleSize = autoSampleSize;
			this.blockSize = blockSize;
		}
	}
}
