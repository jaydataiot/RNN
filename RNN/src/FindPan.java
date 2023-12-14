
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
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
public class FindPan {

    //RNN 
    public static void main(String[] args) throws IOException {
        ArrayList<ArrayList<String>> data = new ArrayList<>();

        InOut iO = new InOut();
        iO.input(data, new ArrayList<>(), "data.xlsx");
        ArrayList<String> dates = new ArrayList<>();
        dates.add(getDate(data.get(0).get(0)));
        for (String s : data.get(0)) {
            if (!getDate(s).equals(dates.get(dates.size() - 1))) {
                dates.add(getDate(s));
            }
        }

        for (int i = 0; i < dates.size() - 4; i++) {
            String d1 = dates.get(i++);
            String d2 = dates.get(i++);
            String d3 = dates.get(i++);
            String d4 = dates.get(i);
            //------------------------------------------------------------------
            ArrayList<ArrayList<String>> a = new ArrayList<>();
            ArrayList<ArrayList<String>> b = new ArrayList<>();
            for (int j = 1; j < data.size(); j++) {
                a.add(new ArrayList<>());
                b.add(new ArrayList<>());
            }

            //------------------------------------------------------------------
            for (int j = 0; j < data.get(0).size(); j++) {
                String temp = getDate(data.get(0).get(j));
                if (d1.equals(temp) || d2.equals(temp)) {
                    for (int k = 1; k < data.size(); k++) {
                        a.get(k - 1).add(data.get(k).get(j));
                    }
                } else if (d3.equals(temp) || d4.equals(temp)) {
                    for (int k = 1; k < data.size(); k++) {
                        b.get(k - 1).add(data.get(k).get(j));
                    }
                }
            }
            //------------------------------------------------------------------

            System.out.println(d1 + " " + d2 + " " + d3 + " " + d4 + " :: " + euclid(getAvg(a), getAvg(b)));
        }

    }

    public static Double euclid(ArrayList<Double> a, ArrayList<Double> b) {

        double e = 0.0;
        for (int i = 0; i < a.size(); i++) {
            e += ((b.get(i) - a.get(i)) * (b.get(i) - a.get(i)));
        }

        return Math.sqrt(e);
    }

    public static ArrayList<Double> getAvg(ArrayList<ArrayList<String>> a) {
        ArrayList<Double> temp = new ArrayList<>();

        for (int i = 0; i < a.size(); i++) {
            double d = 0.0;
            for (int j = 0; j < a.get(i).size(); j++) {
                d += Double.valueOf(a.get(i).get(j));
            }
            temp.add(d / a.get(i).size());
        }

        return temp;
    }

    public static String getDate(String date) {
        String temp = "";
        for (int i = 0; i < date.length(); i++) {
            if (date.charAt(i) != '/') {
                temp += date.charAt(i);
            } else {
                break;
            }
        }
        temp += date.charAt(date.length() - 2);
        temp += date.charAt(date.length() - 1);
        return temp;
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
