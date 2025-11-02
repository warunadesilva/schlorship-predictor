import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.converters.CSVLoader;

public class ScholarshipPredictor extends JFrame {

    private JTextField ageField, incomeField, degreeField, gpaField, pubField;
    private JLabel resultLabel;
    private Classifier classifier;
    private Instances dataStructure;

    public ScholarshipPredictor() {
        setTitle("Scholarship Predictor");
        setSize(400, 450); 
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setLayout(new GridLayout(9, 2, 10, 10));

        add(new JLabel("Age:"));
        ageField = new JTextField();
        add(ageField);

        add(new JLabel("Income:"));
        incomeField = new JTextField();
        add(incomeField);

        add(new JLabel("Degree (1=Special, 0=General):"));
        degreeField = new JTextField();
        add(degreeField);

        add(new JLabel("GPA:"));
        gpaField = new JTextField();
        add(gpaField);

        add(new JLabel("Publications:"));
        pubField = new JTextField();
        add(pubField);

        JButton predictBtn = new JButton("Predict");
        add(predictBtn);
        
        JButton coeffBtn = new JButton("Show Coefficients");
        add(coeffBtn); 
        
        resultLabel = new JLabel("");
        add(resultLabel);

        trainModel();

        predictBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                predictScholarship();
            }
        });
        
        coeffBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                displayCoefficients();
            }
        });

        setVisible(true);
    }

    private void trainModel() {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("scholarship_structurenew.csv"));
            Instances dataset = loader.getDataSet();

            dataset.setClassIndex(dataset.numAttributes() - 1);
            dataStructure = new Instances(dataset, 0);

            // 1. Initialize the Logistic model
            Logistic logisticModel = new Logistic(); 

            // 2. Perform Evaluation (10-fold Cross-Validation)
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(logisticModel, dataset, 10, new java.util.Random(1));
            
            // 3. Print Summary to Terminal (The requested part) 📊
            System.out.println("\n=============================================");
            System.out.println("  Weka Logistic Regression Model Summary");
            System.out.println("=============================================");
            
            // Overall Accuracy and Summary (Correctly Classified Instances, AUC, etc.)
            System.out.println("\n--- Summary (Accuracy, Error Rates, AUC) ---");
            System.out.println(eval.toSummaryString(false));
            
            // Class-Specific Metrics (Precision, Recall, F-Measure)
            System.out.println("\n--- Detailed Class Performance ---");
            System.out.println(eval.toClassDetailsString());
            
            // Confusion Matrix (TP, TN, FP, FN counts)
            System.out.println("\n--- Confusion Matrix ---");
            System.out.println(eval.toMatrixString());
            System.out.println("=============================================\n");

            // 4. Train the final model on the full dataset
            logisticModel.buildClassifier(dataset);
            classifier = logisticModel; // Set the trained model

        } catch (Exception e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error training model: " + e.getMessage(),
                    "Training Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    /**
     * Method to display the Logistic Regression coefficients in a dialog.
     */
    private void displayCoefficients() {
        if (classifier instanceof Logistic) {
            String coefficients = classifier.toString();
            
            // Use a JTextArea inside a JScrollPane for long output
            JTextArea textArea = new JTextArea(coefficients);
            textArea.setEditable(false);
            JScrollPane scrollPane = new JScrollPane(textArea);
            scrollPane.setPreferredSize(new Dimension(500, 300));

            JOptionPane.showMessageDialog(this, scrollPane, "Logistic Regression Coefficients",
                    JOptionPane.INFORMATION_MESSAGE);
        } else {
            JOptionPane.showMessageDialog(this, "Classifier is not a Logistic Regression model.", 
                                          "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void predictScholarship() {
        try {
            double age = Double.parseDouble(ageField.getText());
            double income = Double.parseDouble(incomeField.getText());
            double degree = Double.parseDouble(degreeField.getText());
            double gpa = Double.parseDouble(gpaField.getText());
            double pub = Double.parseDouble(pubField.getText());

            Instance inst = new DenseInstance(6);
            inst.setDataset(dataStructure);
            inst.setValue(0, age);
            inst.setValue(1, income);
            inst.setValue(2, degree);
            inst.setValue(3, gpa);
            inst.setValue(4, pub);

            double pred = classifier.classifyInstance(inst);
            String result = dataStructure.classAttribute().value((int) pred);

            double[] dist = classifier.distributionForInstance(inst);
            resultLabel.setText("Prediction: " + result + " (" +
                    String.format("%.2f", dist[(int) pred] * 100) + "% confidence)");

        } catch (NumberFormatException nf) {
            JOptionPane.showMessageDialog(this, "Enter valid numeric values.", "Input Error",
                    JOptionPane.ERROR_MESSAGE);
        } catch (NullPointerException npe) {
            JOptionPane.showMessageDialog(this, "Model is not trained correctly.", "Model Error",
                    JOptionPane.ERROR_MESSAGE);
        } catch (Exception e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Unexpected error: " + e.getMessage(), "Error",
                    JOptionPane.ERROR_MESSAGE);
        }
    }

    public static void main(String[] args) {
        new ScholarshipPredictor();
    }
}