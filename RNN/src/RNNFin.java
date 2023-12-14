
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
public class RNNFin {

    //RNN 
    public static void main(String[] args) throws IOException {
        ArrayList<ArrayList<Double>> Train = new ArrayList<>();
        input(Train, new ArrayList<>(), "train-before2-Gold.xlsx");
        Train.add(0, new ArrayList<>());
        System.out.println(Train.size());
        for (int i = 0; i < Train.get(1).size(); i++) {
            Train.get(0).add(-1.0);
        }
        System.out.println(Train.get(1).size());
        double lr = 0.007;
        //creates weights U, V, and W
        ArrayList<ArrayList<Double>> W = new ArrayList<>();
        genWeights(W, 4);

        System.out.println("W init :: ");
        print(W);

        System.out.println("Forward :: ");
        ArrayList<Double> Q = new ArrayList<>();
        double v = 0.0, countT = 0.0, count = 0.0;
        while (v < 0.80) {
            for (int i = 1; i < Train.get(0).size() - 1; i++) {

                //input to X
                ArrayList<Double> K1 = new ArrayList<>();
                //input to C
                ArrayList<Double> K2 = new ArrayList<>();
                //bias + K1 + Q
                ArrayList<Double> K3 = new ArrayList<>();
                //bias + K1 + Q
                ArrayList<Double> uT = new ArrayList<>();
                //add bias
                uT.add(-1.0);
                //add X1-X5
                for (int j = 1; j < Train.size(); j++) {
                    uT.add(Train.get(j).get(i));
                    K1.add(Train.get(j).get(i));
                    K2.add(Train.get(j).get(i - 1));
                    K3.add(Train.get(j).get(i + 1));
                }
                if (Q.isEmpty()) {
                    //add C1-C5 init
                    for (int j = 1; j < Train.size(); j++) {
                        uT.add(Train.get(j).get(i - 1));
                    }
                } else {
                    for (int j = 0; j < Q.size(); j++) {
                        uT.add(Q.get(j));
                    }
                    Q.clear();
                }

                //vT holds the product of W X uT
                ArrayList<Double> vT = new ArrayList<>();
                multiplyW(uT, W, vT);

                //input to Y
                ArrayList<Double> Z = new ArrayList<>();
                //input to Q
                ArrayList<Double> E = new ArrayList<>();
                for (int j = 1; j < 6; j++) {
                    Z.add(vT.get(j));
                    E.add(vT.get(j + 5));
                }

                //output of Y
                ArrayList<Double> Y = new ArrayList<>();
                for (Double d : Z) {
                    Y.add(gelu(d));
                }
                //output of Q
                for (Double d : E) {
                    Q.add(gelu(d));
                }

                int n = 0;
                for (int j = 0; j < Y.size(); j++) {
                    if (Math.abs(Math.round(Y.get(j)) - K3.get(j)) > 1.0) {
                        n++;

                    }
                    countT++;
                }

                count += n;

                updateW(W, Y, Q, K1, K2, uT.get(0), lr);

            }
            System.out.println(count / countT);
            v = count / countT;

        }
    }

    public static void updateW(ArrayList<ArrayList<Double>> W, ArrayList<Double> Y,
            ArrayList<Double> Q, ArrayList<Double> K1, ArrayList<Double> K2, Double bias, double lr) {
        /*System.out.println("Y :: " + Y.toString());
        System.out.println("K1 :: " + K1.toString());
        System.out.println("K2 :: " + K2.toString());
        System.out.println("B :: " + bias);
        System.out.println("___________________________________________");*/

        for (int n = 0; n < W.size(); n++) {

            int i = 0;
            if (n == 0) {
                for (int j = 0; j < K1.size(); j++) {
                    double temp = lr * bias * (K1.get(j) - Y.get(j));
                    double old = W.get(n).get(i);
                    W.get(n).remove(i);
                    W.get(n).add(i++, old + temp);

                }
                for (int j = 0; j < K1.size(); j++) {
                    double temp = lr * bias * (K2.get(j) - Q.get(j));
                    double old = W.get(n).get(i);
                    W.get(n).remove(i);
                    W.get(n).add(i++, old + temp);
                }
            } else if (n > 0 && n < 6) {
                for (int j = 0; j < K1.size(); j++) {
                    double temp = lr * K1.get(j) * (K1.get(j) - Y.get(j));
                    double old = W.get(n).get(i);
                    W.get(n).remove(i);
                    W.get(n).add(i++, old + temp);

                }
                for (int j = 0; j < K1.size(); j++) {
                    double temp = lr * K1.get(j) * (K2.get(j) - Q.get(j));
                    double old = W.get(n).get(i);
                    W.get(n).remove(i);
                    W.get(n).add(i++, old + temp);
                }

            } else {
                for (int j = 0; j < K2.size(); j++) {
                    double temp = lr * K2.get(j) * (K1.get(j) - Y.get(j));
                    double old = W.get(n).get(i);
                    W.get(n).remove(i);
                    W.get(n).add(i++, old + temp);

                }
                for (int j = 0; j < K2.size(); j++) {
                    double temp = lr * K2.get(j) * (K2.get(j) - Q.get(j));
                    double old = W.get(n).get(i);
                    W.get(n).remove(i);
                    W.get(n).add(i++, old + temp);
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
            for (int j = 0; j < 10; j++) {
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

    public static double elu(double alph, double x) {
        if (x > 0) {
            return x;
        } else {
            return alph * Math.exp(x) - 1;
        }

    }

    public static double gelu(double x) {

        return 0.5 * x * (1 + erf(x / Math.sqrt(2)));

    }

    public static double erf(double z) {
        double t = 1.0 / (1.0 + 0.5 * Math.abs(z));

        // use Horner's method
        double ans = 1 - t * Math.exp(-z * z - 1.26551223
                + t * (1.00002368
                + t * (0.37409196
                + t * (0.09678418
                + t * (-0.18628806
                + t * (0.27886807
                + t * (-1.13520398
                + t * (1.48851587
                + t * (-0.82215223
                + t * (0.17087277))))))))));
        if (z >= 0) {
            return ans;
        } else {
            return -ans;
        }
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
        File myFile = new File("//Users/jeffreyyoung/Desktop/Dr. H/H_C_Data/" + name);

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
