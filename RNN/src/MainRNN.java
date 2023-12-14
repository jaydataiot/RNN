
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author jeffreyyoung
 */
public class MainRNN {

    //RNN 
    public static void main(String[] args) throws IOException {
        ArrayList<ArrayList<Double>> Train = new ArrayList<>();
        input(Train, new ArrayList<>(), "Data.xlsx");

        double lr = 0.007;
        //creates weights U, V, and W
        ArrayList<ArrayList<Double>> W = new ArrayList<>();
        genWeights(W, 11);

        System.out.println("W init :: ");
        print(W);

        System.out.println("Forward :: ");

        double v = 0.0;
        while (v < 0.75) {
            double acc = 0.0, count = 0.0;
            //desired value fed back
            ArrayList<Double> Q = new ArrayList<>();
            for (int i = 1; i < Train.get(0).size() - 1; i++) {

                ArrayList<Double> Ut = new ArrayList<>();
                ArrayList<Double> T = new ArrayList<>();
                //add bias
                Ut.add(-1.0);
                T.add(-1.0);
                //add x1 - x5
                for (int j = 1; j < Train.size(); j++) {
                    Ut.add(Train.get(j).get(i - 1));
                    T.add(Train.get(j).get(i - 1));
                }

                if (i == 1) {
                    //add initial c1 - c5
                    for (int j = 1; j < Train.size(); j++) {
                        Ut.add(Train.get(j).get(i));
                        T.add(Train.get(j).get(i));
                    }
                } else {
                    //add c1 to c5 for recurrence
                    for (int j = 0; j < Q.size(); j++) {
                        Ut.add(Q.get(j));
                        T.add(Q.get(j));
                    }
                    Q.clear();
                }

                //System.out.println("Ut :: " + Ut.toString());
                ArrayList<Double> Vt = new ArrayList<>();
                multiplyW(Ut, W, Vt);
                //System.out.println("Vt :: " + Vt.toString());

                ArrayList<Double> Z = new ArrayList<>();
                ArrayList<Double> E = new ArrayList<>();
                for (int j = 0; j < 5; j++) {
                    Z.add(Vt.get(j + 1));
                    E.add(Vt.get(j + 6));
                }

                //actual output
                ArrayList<Double> K = new ArrayList<>();
                for (int j = 1; j < Train.size(); j++) {
                    K.add(Train.get(j).get(i + 1));
                }

                //desired values equal K
                ArrayList<Double> Y = new ArrayList<>();

                //System.out.println("Z :: " + Z.toString());
                //System.out.println("E :: " + E.toString());
                for (int j = 0; j < Z.size(); j++) {
                    Y.add(relu(Z.get(j)));
                    Q.add(relu(E.get(j)));
                }

                if (i == Train.get(0).size() - 2) {
                    double n = 0.0;
                    for (int j = 0; j < K.size(); j++) {

                        System.out.println(K.get(j) + " " + Y.get(j));
                        if (K.get(j) == Math.round(Y.get(j))) {
                            n++;
                        }
                    }
                    acc += (n / K.size());
                    count++;
                }
                /*double n = 0.0;
                for (int j = 0; j < K.size(); j++) {

                    //System.out.println(K.get(j) + " " + Y.get(j));
                    if (Math.round(K.get(j)) == Math.round(Y.get(j))) {
                        n++;
                    }
                }
                acc += (n / K.size());
                count++;*/

                updateW(W, Y, K, T, lr);
                //print(W);

            }
            System.out.println("Predicted%: " + acc / count);
            v = acc / count;
        }

    }

    public static void updateW(ArrayList<ArrayList<Double>> W, ArrayList<Double> Y,
            ArrayList<Double> K, ArrayList<Double> Ut, double lr) {
        /*System.out.println("Y :: " + Y.toString());
        System.out.println("K :: " + K.toString());
        System.out.println("Ut :: " + Ut.toString());
        System.out.println("___________________________________________");*/
        for (int n = 0; n < W.size(); n++) {

            for (int i = 0; i < K.size(); i++) {
                for (int j = 0; j < Ut.size(); j++) {
                    double dw = lr * Ut.get(j) * (K.get(i) - Y.get(i));
                    double v = W.get(n).get(j) + dw;

                    W.get(n).remove(j);
                    W.get(n).add(j, v);
                }
            }

        }
    }

    public static double normal(double x, double xmin, double xmax, double l, double u) {
        return ((x - xmin) / (xmax - xmin)) * (u - l) + l;
    }

    public static double unNormal(double y, double xmin, double xmax, double l, double u) {
        //return ((x - xmin) / (xmax - xmin)) * (u - l) + l;
        return (((y - l) / (u - l)) * (xmax - xmin)) + xmin;

    }

    public static ArrayList<ArrayList<Double>> transpose(ArrayList<ArrayList<Double>> W) {

        ArrayList<ArrayList<Double>> trans = new ArrayList<>();
        for (int i = 0; i < W.size(); i++) {
            trans.add(new ArrayList<>());
        }

        for (int j = 0; j < W.get(0).size(); j++) {
            for (int k = 0; k < trans.size(); k++) {
                trans.get(j).add(W.get(k).get(j));
            }
        }

        return trans;
    }

    //1st arg is input vector second is weights
    public static void multiplyW(ArrayList<Double> X, ArrayList<ArrayList<Double>> W,
            ArrayList<Double> update) {
        if (W.size() == X.size()) {
            for (int i = 0; i < W.size(); i++) {
                double sum = 0.0;
                for (int j = 0; j < W.get(i).size(); j++) {
                    sum += (W.get(i).get(j) * X.get(j));
                }
                update.add(sum);
            }
        } else {
            System.out.println("2 Size is not equal!!!");
        }
    }

    public static void genWeights(ArrayList<ArrayList<Double>> U, int size) {
        for (int i = 0; i < size; i++) {
            U.add(new ArrayList<>());
        }

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double n = 1 + (Math.random() * ((9 - 1) + 1));
                U.get(i).add(n);
            }
        }

        for (int i = 0; i < U.size(); i++) {
            double sum = sumCol(U.get(i));
            for (int j = 0; j < U.get(i).size(); j++) {
                double d = U.get(i).get(j);
                U.get(i).remove(j);
                U.get(i).add(j, d / sum);
            }
        }
    }

    public static double sumCol(ArrayList<Double> a) {
        double sum = 0.0;
        for (Double d : a) {
            sum += d;
        }
        return sum;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double relu(double x) {
        return Math.max(0.0, x);
    }

    public static double softMax(double x, ArrayList<Double> y) {
        double sum = 0.0;
        for (Double d : y) {
            sum += Math.exp(d);
        }
        return Math.exp(x) / sum;
    }

    public static double tanH(double d) {
        return Math.tanh(Math.toRadians(d));
    }

    public static double sigmoidInverse(double y) {
        return Math.log(y / (1 - y));
    }

    public static void print(ArrayList<ArrayList<Double>> ar) {
        for (int i = 0; i < ar.get(0).size(); i++) {
            for (int j = 0; j < ar.size(); j++) {

                if (j != ar.size() - 1) {
                    System.out.print(ar.get(j).get(i) + " | ");
                } else {
                    System.out.print(ar.get(j).get(i));
                }
            }
            System.out.println();
        }
        System.out.println("----------------------------------------------------");
    }

    public static void input(ArrayList<ArrayList<Double>> data, ArrayList<String> names, String name) throws IOException {

        ArrayList<ArrayList<XSSFCell>> cells = new ArrayList<>();

        //This pathway must be set!!!!
        File myFile = new File("//Users/jeffreyyoung/Desktop/Dr. H/RNN/" + name);

        FileInputStream fis = null;

        fis = new FileInputStream(myFile);

        XSSFWorkbook wb = null;

        wb = new XSSFWorkbook(fis);

        XSSFSheet sheet = wb.getSheetAt(0);

        XSSFRow row;

        int rows; // No of rows
        rows = sheet.getPhysicalNumberOfRows();

        int cols = 0; // No of columns
        int tmp = 0;

        // This trick ensures that we get the data properly even if it doesn't start from first few rows
        for (int i = 0; i < 10 || i < rows; i++) {
            row = sheet.getRow(i);
            if (row != null) {
                tmp = sheet.getRow(i).getPhysicalNumberOfCells();
                if (tmp > cols) {
                    cols = tmp;
                }
            }
        }

        for (int n = 0; n < cols; n++) {
            cells.add(new ArrayList<>()); //fills arraylists for number of columns
            data.add(new ArrayList<>());
        }

        for (int r = 0; r < rows * 2; r++) { //*2 to fix halfing problem
            row = sheet.getRow(r);
            if (row != null) {
                for (int c = 0; c < cols; c++) {
                    XSSFCell cell = row.getCell((short) c);
                    if (cell != null) {
                        cells.get(c % cols).add(cell);
                    } else {
                        cell = row.createCell(c);
                        cell.setCellValue("null");
                        cells.get(c % cols).add(cell);
                    }
                }
            }
        }

        for (int i = 0; i < cells.size(); i++) {
            names.add(cells.get(i).get(0).toString());
            for (int j = 1; j < cells.get(i).size(); j++) { //adjust to isolate years
                cells.get(i).get(j).setCellType(CellType.NUMERIC); //convert cell to numeric
                data.get(i).add(cells.get(i).get(j).getNumericCellValue()); //convert cell to double and add to arraylist
            }
        }
        //-------------------input data end-------------------------------------
    }

}
