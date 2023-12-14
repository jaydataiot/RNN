
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

/**
 *
 * @author jeffreyyoung
 */
public class InOut {
    
    InOut(){
        System.out.println("InOut Created::");
    }
    
    public void input(ArrayList<ArrayList<String>> data, ArrayList<String> names, String name) throws FileNotFoundException, IOException {

        ArrayList<ArrayList<XSSFCell>> cells = new ArrayList<>();

        //This pathway must be set!!!!
        File myFile = new File("//Users/jeffreyyoung/Desktop/Dr. H/H_C_Data/" + name);

        FileInputStream fis = null;

        fis = new FileInputStream(myFile);

        XSSFWorkbook wb = null;

        wb = new XSSFWorkbook(fis);

        XSSFSheet sheet = wb.getSheetAt(0);

        XSSFRow row;
        //XSSFCell cell = null;

        int rows; // No of rows
        rows = sheet.getPhysicalNumberOfRows();

        System.out.println("rows = " + rows);
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

        System.out.println("cols: " + cols);
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
            System.out.println("Cell " + i + " contain n = : " + cells.get(i).size());
            names.add(cells.get(i).get(0).toString());
            for (int j = 1; j < cells.get(i).size(); j++) { //adjust to isolate years
                if (names.get(i).equals("Date")) {
                    cells.get(i).get(j).setCellType(CellType.NUMERIC);
                    DateFormat df = new SimpleDateFormat("MM/dd/yyyy");
                    Date date = cells.get(i).get(j).getDateCellValue();
                    data.get(i).add(df.format(date));

                } else {
                    cells.get(i).get(j).setCellType(CellType.STRING); //convert cell to numeric
                    data.get(i).add(cells.get(i).get(j).getStringCellValue()); //convert cell to double and add to arraylist
                }
            }
        }
        //-------------------input data end-------------------------------------
    }

    //Method used to output an Excel file
    public void outputToFile(ArrayList<ArrayList<String>> list, String name) throws IOException {

        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet = workbook.createSheet("Java Books");

        int size = list.get(0).size();
        for (int i = 1; i < list.size(); i++) {
            if (size < list.get(i).size()) {
                size = list.get(i).size();
            }
        }

        Object[][] bookData = new Object[size][list.size()];
        for (int i = 0; i < list.size(); i++) {
            for (int j = 0; j < list.get(i).size(); j++) {
                bookData[j][i] = list.get(i).get(j);
            }
        }

        int rowCount = 0;

        for (Object[] aBook : bookData) {
            Row row = sheet.createRow(rowCount++);

            int columnCount = 0;

            int n = 0;
            for (Object field : aBook) {
                Cell cell = row.createCell(columnCount++);

                if (field instanceof String) {
                    cell.setCellValue((String) field);
                } else if (field instanceof Integer) {
                    cell.setCellValue((Integer) field);
                } else if (field instanceof Double) {
                    cell.setCellValue((Double) field);
                }
            }

        }
        try ( FileOutputStream outputStream = new FileOutputStream("//Users/jeffreyyoung/Desktop/ReviewTracker/Last/" + name)) {
            workbook.write(outputStream);
        }
        System.out.println("file was output");
    }
    
}
