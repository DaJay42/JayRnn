package ch.dajay42.rnn;

import java.io.*;
import java.nio.file.*;
import java.util.*;

import ch.dajay42.application.*;
import ch.dajay42.application.config.*;
import ch.dajay42.math.linAlg.*;


public class RnnMain {
	
	private static RnnEncDec<Byte> asciiEncDec;

	private static Rnn currentRnn = null;
	private static String rnnLoadPath = "";
	private static String rnnStorePath = "";
	
	private static String textPath = "";
	private static byte[] text = null;
	private static int textLength = 0;
	private static Matrix[] textEncd = null;
	private static int offset = 0;
	
	
	private static int chunkSize = 32;
	private static int autoSampleFrequency = 100;
	private static int autoSampleSize = 256;
	private static int blockSize = 256*1024;
	
	private final static Map<String, Setting> SETTING_MAP = new HashMap<>(){
		{
			put("chunkSize", new Setting<>(() -> chunkSize, i -> chunkSize = i, Parser.INTEGER_PARSER));
			put("autoSampleFrequency", new Setting<>(() -> autoSampleFrequency, i -> autoSampleFrequency = i, Parser.INTEGER_PARSER));
			put("autoSampleSize", new Setting<>(() -> autoSampleSize, i -> autoSampleSize = i, Parser.INTEGER_PARSER));
			put("blockSize", new Setting<>(() -> blockSize, i -> blockSize = i, Parser.INTEGER_PARSER));
		}
	};
	
	private final static SimpleCLI CLI = new SimpleCLI();
	
	private final static String statusStr = "status";
	
	private final static String loadStr = "load";
	private final static String storeStr = "store";
	private final static String createStr = "create";
	
	private final static String sampleStr = "sample";
	private final static String learnStr = "learn";
	
	private final static String readStr = "read";
	private final static String writeStr = "write";
	
	private final static String temperatureStr = "settemperature";
	private final static String learnRateStr = "setlearnrate";
	
	private final static String settingsFile = "jayrnn.ini";

	
	private static void initEncDec(){
		Set<Byte> charset = new HashSet<>();
		for(byte s = 0; s >= 0; s++){
			charset.add(s);
		}
		asciiEncDec = new RnnEncDec<>(charset);
	}
	
	private static void loadTextBlock(){
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
	
	private static void loadText(String filename){
		System.out.println("Loading text from \""+filename+"\"...");
		try {
			text = Files.readAllBytes(FileSystems.getDefault().getPath(filename));
		} catch (IOException e) {
			System.err.print("Error: Could not load text from file ");
			System.err.print(filename);
			System.err.print(" due to ");
			System.err.print(e.toString());
			System.err.println();
			return;
		}
		
		textLength = text.length;
		offset = 0;
		textPath = filename;
		textEncd = null;
		
		System.out.println("Text file loaded.");
		loadTextBlock();
	}
	
	private static void loadSettings(){
		try {
			ConfigUtil.readFromFile(settingsFile, SETTING_MAP);
		} catch (IOException e) {
			System.out.println("No settings file found.");
		}
	}
	
	private static void storeSettings(String filename){
		if(!filename.endsWith(".ini"))
			filename = filename + ".ini";
		
		try{
			ConfigUtil.writeToFile(filename, SETTING_MAP);
		} catch (IOException e) {
			System.err.print("Error: Could not store Settings in file ");
			System.err.print(filename);
			System.err.print(" due to ");
			System.err.print(e.toString());
			System.err.println();
			
		}
	}
	
	private static void loadRnn(String filename){
		try {
			byte[] bytes = Files.readAllBytes(FileSystems.getDefault().getPath(filename));
			try(ObjectInputStream s = new ObjectInputStream(new ByteArrayInputStream(bytes))){
				Object o = s.readObject();
				currentRnn = (Rnn) o;
			}
		} catch (IOException | ClassNotFoundException e) {
			System.err.print("Error: Could not load RNN from file ");
			System.err.print(filename);
			System.err.print(" due to ");
			System.err.print(e.toString());
			System.err.println();
			return;
		}
		offset = 0;
		rnnLoadPath = filename;
		rnnStorePath = "";

		System.out.println("RNN loaded from '"+rnnLoadPath+"'.");
	}
	
	private static void storeRnn(String filename){
		
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
			System.err.print("Error: Could not store RNN in file ");
			System.err.print(filename);
			System.err.print(" due to ");
			System.err.print(e.toString());
			System.err.println();
			return;
		}
		rnnStorePath = filename;
		System.out.println("RNN stored to '"+rnnStorePath+"'.");
	}
	
	private static void createRnn(int hiddenSize){
		currentRnn = new MinimalRnn(hiddenSize, asciiEncDec.classes);
		offset = 0;
		rnnLoadPath = "";
		rnnStorePath = "";
		System.out.println("Created new RNN.");
	}
	
	private static void learn(int chunks){
		Matrix[] in = new Matrix[chunkSize];
		Matrix[] exout = new Matrix[chunkSize];
		

		System.out.append('\n');
		for(int i = 0; i < chunks; i++){
			
			if((offset % blockSize) + chunkSize > blockSize){
				loadTextBlock();
				Matrix hmat = new ColumnVectorSparse(currentRnn.getHiddenSize());
				currentRnn.setH(hmat);
			}
			
			Matrix last = null;
			for(int j = 0; j < chunkSize; j++){
				int jk = (j + offset) % blockSize;
				in[j] = textEncd[jk];
				exout[j] = textEncd[jk];
				if(j == chunkSize - 1)
					last = textEncd[jk];
			}
			offset += chunkSize;
			offset %= textLength;
			
			
			currentRnn.learn(in, exout, currentRnn.getH());
			currentRnn.step(last);
			
			if(autoSampleFrequency > 0 && i % autoSampleFrequency == 0){
				Matrix[] seed = {asciiEncDec.encode(text[offset])};
				sample(autoSampleSize, seed);
			}
			System.out.append("Loss/step: ").append(Double.toString(currentRnn.getLastLoss()));
			
			System.out.append('\n');
			System.out.flush();
		}

	}
	
	private static void write(String filename, int chars, Matrix[] seed){
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
			System.err.print("Error: Could not write to file ");
			System.err.print(filename);
			System.err.print(" due to ");
			System.err.print(e.toString());
			System.err.println();
			return;
		}
		System.out.println("Wrote "+chars+" characters to '"+filename+"'.");
	}
	
	private static void sample(int chars, Matrix[] seed){
		sample(chars, seed, System.out);
	}
	
	
	private static void sample(int chars, Matrix[] seed, PrintStream out){
		
		out.append('\n');
		out.append("training steps = ");
		out.append(Long.toString(currentRnn.getLearnedSteps()));
		out.append('\n');

		Matrix h = new ColumnVectorDense(currentRnn.getHiddenSize());
		
		List<Matrix> res = currentRnn.sample(h, seed, chars);
		Byte b;
		for(Matrix r : res){
			b = asciiEncDec.decodeMax(r);
			out.write(b);
		}
		
		out.append('\n');
		out.append('\n');

	}
	
	private static void status(){
		StringBuilder builder = new StringBuilder();
		builder.append("RnnMain:");
		builder.append('\n');
		builder.append('\n');
		builder.append(ConfigUtil.prettyPrint(SETTING_MAP, "\n"));
		builder.append('\n');
		builder.append("[RNN]");
		builder.append('\n');
		
		if(currentRnn != null){
			builder.append("size=").append(currentRnn.getHiddenSize());
			builder.append('\n');
			builder.append("steps=").append(currentRnn.getLearnedSteps());
			builder.append('\n');
			builder.append("current loss=").append(currentRnn.getLastLoss());
			builder.append('\n');
			builder.append("temperature=").append(currentRnn.getTemperature());
			builder.append('\n');
			builder.append("learningrate=").append(currentRnn.getLearningRate());
			builder.append('\n');
		}else{
			builder.append("null");
			builder.append('\n');
		}
		
		builder.append('\n');
		builder.append("[Text]");
		builder.append('\n');
		builder.append("path=").append(textPath);
		builder.append('\n');
		builder.append("length=").append(textLength);
		builder.append('\n');
		builder.append('\n');
		System.out.println(builder.toString());
	}
	
	public static void main(String[] args) {
		System.out.println("Loading...");
		
		initEncDec();
		System.out.println("...");
		
		loadSettings();
		System.out.println("...");
		
		CLI.greeting = "RnnMain: Ready.";
		CLI.registerCommmands(
				new CommandGet(SETTING_MAP),
				new CommandSet(SETTING_MAP),
				Command.create(statusStr, "", "Prints the current status of the RNN", (strings) -> status()),
				Command.create(learnStr, "<n>", "Learn from currently loaded text file for <n> chunks.", (strings) -> {
					if(currentRnn != null){
						if(text != null)
							learn(Integer.parseInt(strings[0]));
						else
							System.out.println("Cannot learn: No text loaded.");
					}else{
						System.out.println("Cannot learn: No RNN loaded.");
					}}),
				Command.create(loadStr,"<filename>","Discards the current RNN and loads the one stored in <filename>.", strings -> loadRnn(strings[0])),
				Command.create(createStr, "<h>", "Discards the current RNN and creates an untrained RNN with <h> hidden internal states.", strings -> createRnn(Integer.parseInt(strings[0]))),
				Command.create(sampleStr, "<n> [<chars>]", "Samples and prints <n> characters from the RNN, starting from the seed characters <chars>, or newline.", strings -> {
					if(currentRnn != null){
						int samples = Integer.parseInt(strings[0]);
						String seedStr = (strings.length > 1) ? strings[1] : "\n";
						byte[] seeds = seedStr.getBytes();
						Matrix[] seed = new Matrix[seedStr.length()];
						for(int i = 0; i < seeds.length; i++){
							seed[i] = asciiEncDec.encode(seeds[i]);
						}
						sample(samples, seed);
					}else{
						System.out.println("Cannot sample: No RNN loaded.");
					}}),
				Command.create(storeStr,"<filename>","Stores the current RNN in <filename>.", strings -> {
					if(currentRnn != null){
						storeRnn(strings[0]);
					}else{
						System.out.println("Cannot store RNN: No RNN loaded.");
					}}),
				Command.create(temperatureStr, "<d>", "Sets the current RNN's temperature to <d>.", strings -> {
					if(currentRnn != null){
						double t = Double.parseDouble(strings[0]);
						currentRnn.setTemperature(t);
						System.out.println("Set temperature to "+t+".");
					}else
						System.out.println("Cannot set temperature: no RNN loaded.");
					}),
				Command.create(learnRateStr, "<d>", "Sets the current RNN's learnRate to <d>.", strings -> {
					if(currentRnn != null){
						double r = Double.parseDouble(strings[0]);
						currentRnn.setLearningRate(r);
						System.out.println("Set learning rate to "+r+".");
					}else
						System.out.println("Cannot set learning rate: no RNN loaded.");
				}),
				Command.create(readStr, "<filename>", "Reads the file <filename> and sets it to be used as input for the RNN.", strings -> loadText(strings[0])),
				Command.create(writeStr,"<filename> [<n> [<chars>]]","Samples <n> characters from the RNN, starting from the seed characters <chars>, or newline, writing them to <filename>.", strings -> {
					if(currentRnn != null){
						int chars = (strings.length > 1) ? Integer.parseInt(strings[1]) : autoSampleSize;
						
						String seedStr = (strings.length > 2) ?  strings[2] : "\n\n";
						byte[] seeds = seedStr.getBytes();
						Matrix[] seed = new Matrix[seedStr.length()];
						for(int i = 0; i < seeds.length; i++){
							seed[i] = asciiEncDec.encode(seeds[i]);
						}
						
						write(strings[0], chars, seed);
					}else{
						System.out.println("Cannot write: No RNN loaded.");
					}})
		);
		
		CLI.parseProgramArgs(args);
		
		CLI.greet();
		
		CLI.readContinuously();
		
		storeSettings(settingsFile);
	}
}
